#!/usr/bin/env python3
"""
Ablation Studies (paper-ready)
- Single-GPU (cuda:0), BF16 default
- Exact post-template context control for scaling tests
- Per-run timings; median (p50) & p95 throughput
- Reproducible (seeded), prefill+decode scope
"""

import json
import csv
import time
import math
import argparse
import os
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_single_gpu(device_index: int = 0):
    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.set_device(device_index)
    # Light cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(math.ceil(q * (len(xs) - 1)))
    return xs[idx]


class AblationAnalyzer:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    # ---------- Loading ----------

    def load_model(self, torch_dtype: torch.dtype = torch.bfloat16):
        print(f"Loading {self.model_name} on {self.device} (dtype={torch_dtype})...")
        set_single_gpu(0)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            # Common for decoder-only; use EOS as PAD
            if self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model (force single GPU)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,
        ).to(self.device)
        self.model.eval()
        torch.cuda.synchronize()
        print("Model loaded.")

    # ---------- Prompt helpers ----------

    def format_prompt(self, user_text: str) -> str:
        """Apply chat template if present; else return raw text."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": user_text}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return user_text

    def _long_text(self) -> str:
        return (
            "The following is a comprehensive analysis of artificial intelligence and "
            "machine learning technologies, covering deep learning, NLP, computer vision, "
            "and practical implementations across modern systems. "
        ) * 4096

    def build_inputs_exact(self, target_ctx: int, raw_prompt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Build inputs with EXACT target_ctx tokens *after* applying chat template.
        If raw_prompt is None, generate a long text and slice to length.
        Returns (input_ids, attention_mask, actual_ctx) on self.device.
        """
        assert self.tokenizer is not None

        text = raw_prompt if raw_prompt is not None else self._long_text()
        formatted = self.format_prompt(text)

        enc = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        )
        ids = enc["input_ids"][0]

        # Trim or pad to exact length
        if ids.numel() > target_ctx:
            ids = ids[:target_ctx]
        elif ids.numel() < target_ctx:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            pad_len = target_ctx - ids.numel()
            pad = torch.full((pad_len,), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)

        input_ids = ids.unsqueeze(0).to(self.device)
        attn_mask = torch.ones_like(input_ids, device=self.device)
        return input_ids, attn_mask, int(input_ids.shape[1])

    # ---------- Benchmark primitive ----------

    def _generate_once(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gen_cfg: Dict[str, Any],
        max_new_tokens: int,
    ) -> Tuple[float, int]:
        """Run one generation and return (elapsed_seconds, new_tokens)."""
        assert self.model is not None and self.tokenizer is not None
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                **gen_cfg,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0

        # Count tokens actually generated (may stop early)
        new_tokens = int(out.shape[1] - input_ids.shape[1])
        return dt, new_tokens

    def benchmark_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gen_cfg: Dict[str, Any],
        num_runs: int,
        max_new_tokens: int,
        seed_base: int = 1234,
    ) -> Dict[str, Any]:
        """Run multiple generations; return robust stats."""
        # Warm-up (short)
        with torch.inference_mode():
            _ = self.model.generate(
                input_ids=input_ids[:, : min(16, input_ids.size(1))],
                attention_mask=attention_mask[:, : min(16, attention_mask.size(1))],
                max_new_tokens=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                do_sample=bool(gen_cfg.get("do_sample", False)),
                temperature=float(gen_cfg.get("temperature", 1.0)),
                top_p=float(gen_cfg.get("top_p", 1.0)),
                top_k=int(gen_cfg.get("top_k", 0)),
            )
        torch.cuda.synchronize()

        times, toks, tpots = [], [], []
        for r in range(num_runs):
            # Reproducible sampling
            torch.manual_seed(seed_base + r)
            torch.cuda.manual_seed_all(seed_base + r)

            dt, new_toks = self._generate_once(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gen_cfg=gen_cfg,
                max_new_tokens=max_new_tokens,
            )
            times.append(dt)
            toks.append(new_toks)
            tpots.append((new_toks / dt) if dt > 0 else 0.0)

            # Small rest to reduce thermal drift
            time.sleep(0.15)

        # Stats
        p50_tpot = percentile(tpots, 0.50)
        p95_tpot = percentile(tpots, 0.95)
        mean_tpot = sum(tpots) / max(1, len(tpots))

        p50_time = percentile(times, 0.50)
        p95_time = percentile(times, 0.95)
        mean_time = sum(times) / max(1, len(times))

        p50_new = percentile(toks, 0.50)
        mean_new = sum(toks) / max(1, len(toks))

        return {
            "throughput_scope": "prefill+decode",
            "p50_tokps": p50_tpot,
            "p95_tokps": p95_tpot,
            "mean_tokps": mean_tpot,
            "p50_time_s": p50_time,
            "p95_time_s": p95_time,
            "mean_time_s": mean_time,
            "p50_new_tokens": p50_new,
            "mean_new_tokens": mean_new,
            "runs": len(times),
        }

    # ---------- Ablations ----------

    def test_decoding_parameters(
        self, num_runs: int, max_new_tokens: int
    ) -> List[Dict[str, Any]]:
        print("Decoding parameters…")
        prompt = "Explain the benefits of renewable energy in 2–3 sentences."
        formatted = self.format_prompt(prompt)
        enc = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        configs = [
            ({"do_sample": False}, "Greedy"),
            ({"do_sample": True, "temperature": 0.7, "top_p": 0.9}, "Top-p (0.9)"),
            ({"do_sample": True, "temperature": 0.7, "top_k": 50}, "Top-k (50)"),
            ({"do_sample": True, "temperature": 1.0, "top_p": 0.95}, "High Temp"),
            ({"do_sample": True, "temperature": 0.3, "top_p": 0.8}, "Low Temp"),
        ]

        rows = []
        for gen_cfg, name in configs:
            stats = self.benchmark_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gen_cfg=gen_cfg,
                num_runs=num_runs,
                max_new_tokens=max_new_tokens,
            )
            row = {
                "model": self.model_name,
                "decoding_method": name,
                "max_new_tokens": max_new_tokens,
                **stats,
            }
            rows.append(row)
            print(f"  {name:<12} p50={row['p50_tokps']:.2f} tok/s, mean={row['mean_tokps']:.2f}")
        return rows

    def test_context_length_scaling(
        self, context_lengths: List[int], num_runs: int, max_new_tokens: int
    ) -> List[Dict[str, Any]]:
        print("Context length scaling…")
        rows = []
        # Respect model context limit if present
        max_ctx = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)

        for target_ctx in context_lengths:
            if max_ctx is not None and target_ctx > int(max_ctx):
                print(f"  Skipping ctx={target_ctx} (exceeds model max {max_ctx})")
                continue

            # Build exact post-template ctx
            input_ids, attention_mask, actual_ctx = self.build_inputs_exact(target_ctx)
            stats = self.benchmark_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gen_cfg={"do_sample": False},
                num_runs=num_runs,
                max_new_tokens=max_new_tokens,
            )
            row = {
                "model": self.model_name,
                "target_context_length": int(target_ctx),
                "measured_context_tokens": int(actual_ctx),
                "max_new_tokens": max_new_tokens,
                **stats,
            }
            rows.append(row)
            print(f"  ctx={target_ctx:4d} p50={row['p50_tokps']:.2f} tok/s (measured={actual_ctx})")
        return rows

    def test_quantization_effects(
        self, num_runs: int, max_new_tokens: int
    ) -> List[Dict[str, Any]]:
        print("Quantization effects…")
        base_prompt = "Write a short poem about artificial intelligence."
        formatted = self.format_prompt(base_prompt)
        enc = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        base_input = enc["input_ids"].to(self.device)
        base_mask = enc.get("attention_mask", torch.ones_like(base_input)).to(self.device)

        precisions = [
            (torch.bfloat16, "BF16"),
            (torch.float16,  "FP16"),
            (torch.float32,  "FP32"),
        ]

        rows = []
        for dtype, name in precisions:
            print(f"  Testing {name}…")
            try:
                # Reload model with requested dtype
                del self.model
                torch.cuda.empty_cache()
                self.load_model(torch_dtype=dtype)

                stats = self.benchmark_generation(
                    input_ids=base_input,
                    attention_mask=base_mask,
                    gen_cfg={"do_sample": False},
                    num_runs=num_runs,
                    max_new_tokens=max_new_tokens,
                )
                row = {"model": self.model_name, "precision": name, "max_new_tokens": max_new_tokens, **stats}
            except Exception as e:
                row = {
                    "model": self.model_name,
                    "precision": name,
                    "max_new_tokens": max_new_tokens,
                    "error": str(e),
                }
                print(f"    -> {name} failed: {e}")
            rows.append(row)

        # Restore default BF16 model for any subsequent tests
        del self.model
        torch.cuda.empty_cache()
        self.load_model(torch_dtype=torch.bfloat16)
        return rows

    def test_server_comparison(
        self, num_runs: int, max_new_tokens: int, include_vllm_placeholder: bool = False
    ) -> List[Dict[str, Any]]:
        print("Server framework comparison…")
        prompt = "Explain quantum computing in simple terms."
        formatted = self.format_prompt(prompt)
        enc = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        # Transformers (local)
        stats = self.benchmark_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            gen_cfg={"do_sample": False},
            num_runs=num_runs,
            max_new_tokens=max_new_tokens,
        )
        rows = [{
            "framework": "Transformers",
            "model": self.model_name,
            "max_new_tokens": max_new_tokens,
            **stats,
        }]

        # Optional placeholder (off by default)
        if include_vllm_placeholder:
            rows.append({
                "framework": "vLLM",
                "model": self.model_name,
                "note": "Not measured in this script; requires separate server.",
            })
        return rows

    # ---------- Orchestration ----------

    def run(
        self,
        tests: List[str],
        context_lengths: List[int],
        num_runs: int,
        max_new_tokens: int,
        include_vllm_placeholder: bool,
    ) -> Dict[str, List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        if "decoding" in tests or "all" in tests:
            results["decoding_parameters"] = self.test_decoding_parameters(num_runs, max_new_tokens)
        if "quantization" in tests or "all" in tests:
            results["quantization_effects"] = self.test_quantization_effects(num_runs, max_new_tokens)
        if "context" in tests or "all" in tests:
            results["context_scaling"] = self.test_context_length_scaling(context_lengths, num_runs, max_new_tokens)
        if "server" in tests or "all" in tests:
            results["server_comparison"] = self.test_server_comparison(num_runs, max_new_tokens, include_vllm_placeholder)
        return results

    # ---------- Output ----------

    def save_results(self, results: Dict[str, List[Dict[str, Any]]], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        tag = self.model_name.replace("/", "_")

        # CSVs
        for key, rows in results.items():
            if not rows:
                continue
            # Collect union of fields to avoid missing columns
            fieldnames = sorted({k for r in rows for k in r.keys()})
            path = os.path.join(output_dir, f"ablation_{key}_{tag}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

        # JSON
        json_path = os.path.join(output_dir, f"ablation_all_{tag}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Ablation results saved to {output_dir}")

    def print_summary(self, results: Dict[str, List[Dict[str, Any]]]):
        print("\n" + "=" * 100)
        print(f"ABLATION STUDIES SUMMARY — {self.model_name}")
        print("=" * 100)

        def quick_line(label: str, rows: List[Dict[str, Any]], sort_key: str = "p50_tokps"):
            if not rows:
                return
            print(f"\n{label}:")
            print("-" * 80)
            for r in rows:
                parts = []
                if "decoding_method" in r:
                    parts.append(f"{r['decoding_method']:<14}")
                if "precision" in r:
                    parts.append(f"{r['precision']:<6}")
                if "target_context_length" in r:
                    parts.append(f"ctx={r['target_context_length']:<5} (meas={r.get('measured_context_tokens','?')})")
                if "framework" in r:
                    parts.append(f"{r['framework']:<12}")
                parts.append(f"p50={r.get('p50_tokps', 0):>6.2f} tok/s")
                parts.append(f"mean={r.get('mean_tokps', 0):>6.2f}")
                print("  " + "  ".join(parts))

        quick_line("Decoding parameters", results.get("decoding_parameters", []))
        quick_line("Quantization effects", results.get("quantization_effects", []))
        quick_line("Context scaling", results.get("context_scaling", []))
        quick_line("Server comparison", results.get("server_comparison", []))
        print("\n" + "=" * 100)


def parse_args():
    ap = argparse.ArgumentParser(description="Run paper-ready ablation studies")
    ap.add_argument("--model", required=True, help="Model name/path (e.g., openai/gpt-oss-20b)")
    ap.add_argument("--output_dir", default="results/ablations", help="Output directory")
    ap.add_argument("--tests", default="all", help="decoding,quantization,context,server,all")
    ap.add_argument("--context_lengths", default="512,1024,2048,4096", help="Comma-separated target contexts")
    ap.add_argument("--num_runs", type=int, default=5, help="Runs per measurement")
    ap.add_argument("--max_new_tokens", type=int, default=64, help="Decode length")
    ap.add_argument("--include_vllm_placeholder", action="store_true", help="Add vLLM 'not measured' row")
    return ap.parse_args()


def main():
    args = parse_args()
    analyzer = AblationAnalyzer(args.model)
    analyzer.load_model(torch_dtype=torch.bfloat16)

    tests = [t.strip().lower() for t in args.tests.split(",")]
    ctxs = [int(x.strip()) for x in args.context_lengths.split(",") if x.strip()]

    results = analyzer.run(
        tests=tests,
        context_lengths=ctxs,
        num_runs=args.num_runs,
        max_new_tokens=args.max_new_tokens,
        include_vllm_placeholder=args.include_vllm_placeholder,
    )

    analyzer.save_results(results, args.output_dir)
    analyzer.print_summary(results)
    print("Done.")


if __name__ == "__main__":
    main()
