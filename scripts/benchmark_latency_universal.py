#!/usr/bin/env python3
"""
Universal Latency Benchmark (paper-ready)
- Single-GPU, bf16
- Post-template context length is EXACT (trim/pad)
- True TTFT = 1-token generation time (includes prefill), median of 5
- E2E latency percentiles over N runs
- TPOT = generated tok/sec over the full decode (median)
"""

import argparse
import csv
import math
import statistics
import time
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class UniversalLatencyBenchmark:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_type = None

    # ---------- Setup ----------

    def load_model(self):
        """Load model on a single GPU (no sharding) and prep tokenizer."""
        print(f"Loading {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,              # single GPU for apples-to-apples
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Ensure padding is defined (fallback to EOS)
        if self.tokenizer.pad_token_id is None:
            # Many chat models only define EOS; set pad_token to EOS for benchmarking
            if self.tokenizer.eos_token is None:
                # Last resort (should be rare), create a dummy token
                self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_type = self._determine_model_format()
        print(f"Model loaded. Format: {self.model_type}")

    def _determine_model_format(self) -> str:
        """Heuristic for prompt formatting."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return "chat_template"
        name = (self.model_name or "").lower()
        if "instruct" in name or "chat" in name:
            return "instruction"
        return "completion"

    # ---------- Prompt + inputs ----------

    def format_prompt(self, prompt: str) -> str:
        """Format prompt according to model type for fair comparison."""
        if self.model_type == "chat_template":
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif self.model_type == "instruction":
            return f"### Instruction:\n{prompt}\n\n### Response:\n"
        else:
            return prompt

    def generate_prompt(self, target_tokens: int) -> str:
        """Create a long raw text; exact ctx is enforced *after* templating."""
        base = (
            "The following is a comprehensive analysis of artificial intelligence and machine learning technologies. "
            "We will explore various aspects including deep learning architectures, natural language processing, "
            "computer vision, and their applications in modern systems. The discussion will cover both theoretical "
            "foundations and practical implementations, examining how these technologies are transforming industries "
            "and society as a whole."
        )
        # Rough repeat to exceed target; we will trim/pad post-templating
        # Use tokenizer length if you want to be fancy; not required since we trim later.
        reps = max(1, target_tokens // max(1, len(base.split())))
        return (" " + base) * reps

    def build_inputs_exact(self, target_ctx: int, raw_prompt: str) -> torch.Tensor:
        """
        Apply template, then trim/pad to EXACT target_ctx tokens (post-template).
        Ensures every model sees the same context length.
        """
        formatted = self.format_prompt(raw_prompt)
        enc = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False
        )
        ids = enc["input_ids"][0]
        if ids.numel() > target_ctx:
            ids = ids[:target_ctx]
        elif ids.numel() < target_ctx:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            pad_len = target_ctx - ids.numel()
            pad = torch.full((pad_len,), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
        return ids.unsqueeze(0).to(self.model.device)

    # ---------- Timing primitives ----------

    def measure_ttft_ms(self, input_ids: torch.Tensor) -> float:
        """True TTFT: time for a *single* new token (includes prefill)."""
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            _ = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        return (time.time() - t0) * 1000.0

    def decode_once(self, input_ids: torch.Tensor, max_new_tokens: int):
        """One decode to get total wall time (ms), #generated tokens, TPOT (tok/s)."""
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False  # returns Tensor
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        gen = out.shape[1] - input_ids.shape[1]
        tpot = (gen / dt) if dt > 0 else 0.0
        return dt * 1000.0, gen, tpot  # ms, tokens, tok/s

    # ---------- Quantiles ----------

    @staticmethod
    def qtile(vals: List[float], q: float) -> float:
        """Quantile with ceil index (inclusive)."""
        if not vals:
            return 0.0
        xs = sorted(vals)
        n = len(xs)
        idx = int(math.ceil(q * (n - 1)))
        return xs[idx]

    # ---------- Benchmark loop ----------

    def run_benchmark(
        self,
        prompt_tokens_list: List[int],
        gen_tokens_list: List[int],
        num_requests: int = 10
    ) -> List[Dict[str, Any]]:
        results = []

        for prompt_tokens in prompt_tokens_list:
            for gen_tokens in gen_tokens_list:
                print(f"Benchmarking: ctx={prompt_tokens}, gen={gen_tokens}")

                # Build exact post-template context
                raw = self.generate_prompt(prompt_tokens)
                input_ids = self.build_inputs_exact(prompt_tokens, raw)

                # Warm-up once (short)
                with torch.inference_mode():
                    _ = self.model.generate(
                        input_ids=input_ids[:, :min(16, input_ids.size(1))],
                        max_new_tokens=1,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                torch.cuda.synchronize()

                totals_ms, tpots = [], []
                for i in range(num_requests):
                    if i % 5 == 0:
                        print(f"  Request {i+1}/{num_requests}")
                    total_ms, gen_out, tpot = self.decode_once(input_ids, gen_tokens)
                    totals_ms.append(total_ms)
                    tpots.append(tpot)

                # True TTFT (median of 5)
                ttfts = [self.measure_ttft_ms(input_ids) for _ in range(5)]
                ttft_ms = statistics.median(ttfts)

                p50 = statistics.median(totals_ms)
                p95 = self.qtile(totals_ms, 0.95)
                p99 = self.qtile(totals_ms, 0.99)
                tpot_med = statistics.median(tpots)

                results.append({
                    "model": self.model_name,
                    "model_type": self.model_type,
                    "prompt_tokens": prompt_tokens,
                    "gen_tokens": gen_tokens,
                    "TTFT_ms": ttft_ms,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "TPOT_tokps": tpot_med
                })

        return results


# ---------- IO helpers ----------

def save_results(results: List[Dict[str, Any]], output_path: str):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "model", "model_type", "prompt_tokens", "gen_tokens",
            "TTFT_ms", "p50_ms", "p95_ms", "p99_ms", "TPOT_tokps"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)


def print_summary(results: List[Dict[str, Any]]):
    print("\nBenchmark Summary:")
    for r in results:
        print(
            f"  {r['model']}  ctx={r['prompt_tokens']}, gen={r['gen_tokens']}  "
            f"TTFT={r['TTFT_ms']:.1f} ms | p50={r['p50_ms']:.1f} ms | "
            f"TPOT={r['TPOT_tokps']:.1f} tok/s"
        )


def main():
    parser = argparse.ArgumentParser(description="Universal Latency Benchmark (paper-ready)")
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--prompt_tokens", default="128,512,1024,2048",
                        help="Comma-separated post-template context lengths")
    parser.add_argument("--gen_tokens", default="64,128,256",
                        help="Comma-separated generation token counts")
    parser.add_argument("--num_requests", type=int, default=10,
                        help="Number of requests per (ctx,gen) pair")

    args = parser.parse_args()

    # Repro (minor): fix seeds for CUDA clocks/jitter a bit
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    prompt_tokens_list = [int(x.strip()) for x in args.prompt_tokens.split(",") if x.strip()]
    gen_tokens_list = [int(x.strip()) for x in args.gen_tokens.split(",") if x.strip()]

    bench = UniversalLatencyBenchmark(args.model, device="cuda")
    bench.load_model()

    results = bench.run_benchmark(prompt_tokens_list, gen_tokens_list, args.num_requests)
    save_results(results, args.output)
    print(f"\nSaved: {args.output}")
    print_summary(results)


if __name__ == "__main__":
    main()
