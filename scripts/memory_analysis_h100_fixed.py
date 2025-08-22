#!/usr/bin/env python3
"""
H100 VRAM/KV scaling measurement (V2 - stable, single-GPU)
- Exact post-template context control (chat templates supported)
- Reports current, reserved, and *peak* allocator stats
- KV cache MB estimated from peak - post-tokenize (allocated & reserved)
- Keeps PKV alive during measurement (manual prefill+decode)
"""

import os, csv, argparse, gc, math
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MB = 1024 * 1024

def mb(x: int) -> float:
    return float(x) / MB

class H100MemoryAnalyzerV2:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_type = "completion"  # "chat_template" | "instruction" | "completion"

    # ---------- setup ----------
    def load_model(self):
        assert torch.cuda.is_available(), "CUDA not available"
        torch.cuda.set_device(0)  # H100 index 0
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # single GPU, bf16 for H100
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # ensure pad/eos are defined (some chat models only define eos)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_type = self._determine_model_format()

        torch.cuda.synchronize()
        print(f"Loaded {self.model_name} on {self.device} ({torch.cuda.get_device_name(0)})")
        print(f"Detected model format: {self.model_type}")

    def _determine_model_format(self) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            tmpl = (self.tokenizer.chat_template or "").strip()
            if tmpl:
                return "chat_template"
        name = (self.model_name or "").lower()
        if "instruct" in name or "chat" in name:
            return "instruction"
        return "completion"

    # ---------- prompt + exact ctx ----------
    def _base_text(self) -> str:
        return (
            "The following is a comprehensive analysis of artificial intelligence and machine learning technologies. "
            "We will explore deep learning architectures, natural language processing, computer vision, and practical "
            "implementations across modern systems, with attention to theory and applications."
        )

    def _format_prompt(self, prompt: str) -> str:
        if self.model_type == "chat_template":
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif self.model_type == "instruction":
            return f"### Instruction:\n{prompt}\n\n### Response:\n"
        else:
            return prompt

    def _gen_long_prompt(self, target_tokens_hint: int) -> str:
        # Just make it long; exact length happens post-templating
        base = self._base_text()
        # rough repetition by words to exceed target
        reps = max(1, math.ceil(target_tokens_hint / max(1, len(base.split()))))
        return (" " + base) * reps

    def build_inputs_exact(self, target_ctx: int) -> Dict[str, torch.Tensor]:
        """Apply chat/instruction template first, then trim/pad to EXACT length."""
        raw = self._gen_long_prompt(target_ctx)
        formatted = self._format_prompt(raw)
        enc = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"][0]

        # enforce exact length
        if ids.numel() > target_ctx:
            ids = ids[:target_ctx]
        elif ids.numel() < target_ctx:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            pad = torch.full((target_ctx - ids.numel(),), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)

        input_ids = ids.unsqueeze(0).to(self.device)
        attn_mask = torch.ones_like(input_ids, device=self.device)
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    # ---------- measurement primitives ----------
    @staticmethod
    def _check_ctx_limit(model, context_length: int, gen_tokens: int):
        max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if max_ctx is not None and (context_length + gen_tokens) > max_ctx:
            raise ValueError(
                f"(ctx + gen) {context_length + gen_tokens} exceeds model max_position_embeddings {max_ctx}"
            )

    def _warmup(self):
        # tiny warmup to stabilize kernels / cudnn heuristics
        with torch.inference_mode():
            dummy = torch.full((1, 16), self.tokenizer.pad_token_id, device=self.device)
            _ = self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy), use_cache=True)
        torch.cuda.synchronize()

    def _prefill_then_decode_manual(self, batch: Dict[str, torch.Tensor], gen_tokens: int):
        """
        Prefill with full context to allocate KV, then step-decode gen_tokens.
        Keeps a reference to PKV so it stays alive while we read allocator stats.
        """
        with torch.inference_mode():
            pre = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                return_dict=True,
            )
            pkv = pre.past_key_values
            last = batch["input_ids"][:, -1:]

            for _ in range(gen_tokens):
                out = self.model(
                    input_ids=last,
                    past_key_values=pkv,
                    use_cache=True,
                    return_dict=True,
                )
                pkv = out.past_key_values
                # greedy step (we don't care about content, just to extend KV)
                last = out.logits[:, -1:].argmax(dim=-1)

            # keep pkv referenced by returning it
            return pkv

    # ---------- single run ----------
    def _measure_once(self, context_length: int, gen_tokens: int) -> Dict[str, Any]:
        self._check_ctx_limit(self.model, context_length, gen_tokens)

        # Clean allocator state
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        baseline_alloc = mb(torch.cuda.memory_allocated())
        baseline_rsrv  = mb(torch.cuda.memory_reserved())

        # Build exact post-template inputs
        batch = self.build_inputs_exact(context_length)
        torch.cuda.synchronize()
        after_tok_alloc = mb(torch.cuda.memory_allocated())
        after_tok_rsrv  = mb(torch.cuda.memory_reserved())

        # Warm-up, then clear peaks so we only capture the measurement's peaks
        self._warmup()
        torch.cuda.reset_peak_memory_stats()

        # Prefill + manual decode to *keep* PKV alive while we inspect memory
        pkv = self._prefill_then_decode_manual(batch, gen_tokens)
        torch.cuda.synchronize()

        # Capture current + peak with PKV still referenced
        after_gen_alloc = mb(torch.cuda.memory_allocated())
        after_gen_rsrv  = mb(torch.cuda.memory_reserved())
        peak_alloc      = mb(torch.cuda.max_memory_allocated())
        peak_rsrv       = mb(torch.cuda.max_memory_reserved())

        # Drop references (after reading allocator stats)
        del pkv
        torch.cuda.synchronize()

        # Deltas (allocator "current" view)
        tokenize_delta_alloc = after_tok_alloc - baseline_alloc
        generate_delta_alloc = after_gen_alloc - after_tok_alloc
        total_delta_alloc    = after_gen_alloc - baseline_alloc

        # Peak-based KV estimates
        kv_alloc_mb = max(0.0, peak_alloc - after_tok_alloc)
        kv_rsrv_mb  = max(0.0, peak_rsrv  - after_tok_rsrv)

        total_gpu_mb = mb(torch.cuda.get_device_properties(0).total_memory)
        util_after_alloc = (after_gen_alloc / total_gpu_mb) * 100.0
        util_peak_alloc  = (peak_alloc / total_gpu_mb) * 100.0
        util_peak_rsrv   = (peak_rsrv  / total_gpu_mb) * 100.0

        total_tokens = context_length + gen_tokens
        mem_per_token_alloc_mb = (kv_alloc_mb / total_tokens) if total_tokens else 0.0
        mem_per_token_rsrv_mb  = (kv_rsrv_mb  / total_tokens) if total_tokens else 0.0

        return {
            "model": self.model_name,
            "device_name": torch.cuda.get_device_name(0),
            "context_length": context_length,
            "gen_tokens": gen_tokens,
            "total_tokens": total_tokens,

            "baseline_alloc_mb": baseline_alloc,
            "baseline_reserved_mb": baseline_rsrv,
            "after_tokenize_alloc_mb": after_tok_alloc,
            "after_tokenize_reserved_mb": after_tok_rsrv,
            "after_generate_alloc_mb": after_gen_alloc,
            "after_generate_reserved_mb": after_gen_rsrv,
            "peak_alloc_mb": peak_alloc,
            "peak_reserved_mb": peak_rsrv,

            "tokenize_delta_alloc_mb": tokenize_delta_alloc,
            "generate_delta_alloc_mb": generate_delta_alloc,
            "total_delta_alloc_mb": total_delta_alloc,

            # KV estimates (peak-based)
            "kv_cache_alloc_mb": kv_alloc_mb,
            "kv_cache_reserved_mb": kv_rsrv_mb,
            "memory_per_token_alloc_mb": mem_per_token_alloc_mb,
            "memory_per_token_reserved_mb": mem_per_token_rsrv_mb,

            "util_after_percent_alloc": util_after_alloc,
            "util_peak_percent_alloc": util_peak_alloc,
            "util_peak_percent_reserved": util_peak_rsrv,

            "total_gpu_memory_mb": total_gpu_mb,
        }

    # ---------- sweep ----------
    def run(self, ctx_list: List[int], gen_tokens: int) -> List[Dict[str, Any]]:
        results = []
        for ctx in ctx_list:
            try:
                res = self._measure_once(ctx, gen_tokens)
                results.append(res)
                print(
                    f"ctx={ctx:5d} | KV≈ {res['kv_cache_alloc_mb']:.1f} MB "
                    f"(alloc), {res['kv_cache_reserved_mb']:.1f} MB (reserv) | "
                    f"per-token≈ {res['memory_per_token_alloc_mb']:.3f} MB "
                    f"(alloc) | peak_alloc={res['peak_alloc_mb']:.1f} MB"
                )
            except Exception as e:
                print(f"[warn] ctx {ctx}: {e}")
        return results

    # ---------- IO ----------
    @staticmethod
    def save_csv(rows: List[Dict[str, Any]], out_path: str):
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not rows:
            return
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description="H100 VRAM / KV scaling analyzer (V2)")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--context_lengths", default="128,512,1024,2048")
    p.add_argument("--gen_tokens", type=int, default=64)
    args = p.parse_args()

    ctxs = [int(x.strip()) for x in args.context_lengths.split(",") if x.strip()]

    analyzer = H100MemoryAnalyzerV2(args.model)
    analyzer.load_model()
    rows = analyzer.run(ctxs, args.gen_tokens)
    analyzer.save_csv(rows, args.output)

    # quick summary
    if rows:
        print("\n=== SUMMARY (V2) ===")
        print(f"{'ctx':>6}  {'KV_alloc(MB)':>12}  {'KV_resv(MB)':>12}  "
              f"{'per_tok_alloc(MB)':>16}  {'peak_alloc(MB)':>14}")
        for r in rows:
            print(
                f"{r['context_length']:6d}  "
                f"{r['kv_cache_alloc_mb']:12.1f}  "
                f"{r['kv_cache_reserved_mb']:12.1f}  "
                f"{r['memory_per_token_alloc_mb']:16.3f}  "
                f"{r['peak_alloc_mb']:14.1f}"
            )

if __name__ == "__main__":
    # Extra safety: inference-only everywhere
    torch.set_grad_enabled(False)
    main()
