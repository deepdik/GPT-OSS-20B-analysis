#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Memory Benchmark (Fixed, publication-ready)

What this does:
- Loads a HF model on one CUDA device in BF16 (configurable).
- Builds prompts that *approximately* hit a target token count **after** chat templating (if available).
- For each (prompt_len, gen_len) it records:
    * baseline_alloc_mb      : memory with model loaded (weights etc.)
    * peak_alloc_mb          : absolute peak allocated memory during decoding
    * kv_cache_alloc_mb      : estimated KV = slope(MB/token) * total_tokens
    * memory_per_token_alloc_mb : slope (MB/token) from a linear fit on peaks vs total_tokens
    * utilization_peak_percent  : 100 * peak_alloc_mb / total_gpu_memory_mb
    * total_gpu_memory_mb

How KV is estimated (robust):
- For a fixed prompt, we run gen_len ∈ {64, 128, 256} (configurable).
- We record absolute peak allocated MB for each run.
- We fit: peak_MB = intercept + slope * total_tokens  (least squares)
- slope = MB per token (≈ KV per token).  KV(total) ≈ slope * total_tokens.
  This removes most ephemeral operator scratch/fragmentation via the intercept.

Fairness:
- If the tokenizer provides a chat template, we use it; otherwise we use raw completion text.
- Prompt lengths are the *actual tokenized lengths*, reported in the CSV.

Notes:
- Uses torch.cuda.max_memory_allocated() (allocated), not reserved().
- Between runs: gc.collect() + torch.cuda.empty_cache() and reset_peak_memory_stats().

Requires: transformers>=4.40, torch>=2.1
"""

import argparse
import csv
import gc
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def mib(bytes_val: int) -> float:
    return bytes_val / (1024.0 * 1024.0)


@dataclass
class RunPoint:
    total_tokens: int
    peak_alloc_mb: float


class MemoryBench:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self.model_type = None  # "chat_template" | "instruction" | "completion"

    # ------------------------- Model / tokenizer -------------------------

    def _torch_dtype(self):
        if self.dtype.lower() in ["bf16", "bfloat16"]:
            return torch.bfloat16
        if self.dtype.lower() in ["fp16", "float16", "half"]:
            return torch.float16
        return torch.float32

    def load(self):
        print(f"[load] Loading {self.model_name} on {self.device} ({self.dtype})...")
        torch.cuda.set_device(self.device.split(":")[-1])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._torch_dtype(),
            low_cpu_mem_usage=True,
            trust_remote_code=self.trust_remote_code,
            device_map={ "": self.device },  # single device for clean accounting
        )
        # ensure cache is on
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        # determine format
        if getattr(self.tokenizer, "chat_template", None):
            self.model_type = "chat_template"
        elif any(k in self.model_name.lower() for k in ["instruct", "chat"]):
            self.model_type = "instruction"
        else:
            self.model_type = "completion"

        print(f"[load] Format: {self.model_type}")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)

    # ------------------------- Prompt construction -------------------------

    def _apply_template(self, content: str) -> str:
        if self.model_type == "chat_template":
            messages = [{"role": "user", "content": content}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif self.model_type == "instruction":
            return f"### Instruction:\n{content}\n\n### Response:\n"
        else:
            return content

    def _tok_len_after_template(self, content: str) -> int:
        text = self._apply_template(content)
        return len(self.tokenizer(text).input_ids)

    def build_user_content_for_target_tokens(
        self, target_tokens: int, tolerance: int = 2
    ) -> Tuple[str, int]:
        """
        Binary-search the amount of filler to reach ~target tokens AFTER template.
        We use a base paragraph + ' a' repeats (works reliably across BPEs).
        """
        base = (
            "We will analyze model memory behavior under varying context lengths "
            "and generation budgets. This is filler used to control token count."
        )
        # Low/high bounds for number of ' a' repeats
        low, high = 0, 20000  # large upper bound to cover big contexts
        best_text, best_len = base, self._tok_len_after_template(base)

        # If base already exceeds, trim by slicing the base
        if best_len > target_tokens:
            # crude trim
            while best_len > target_tokens and len(base) > 10:
                base = base[:-10]
                best_len = self._tok_len_after_template(base)
                best_text = base
            return best_text, best_len

        # binary search on filler repeats
        while low <= high:
            mid = (low + high) // 2
            content = base + (" a" * mid)
            L = self._tok_len_after_template(content)
            if abs(L - target_tokens) <= tolerance:
                return content, L
            if L < target_tokens:
                best_text, best_len = content, L
                low = mid + 1
            else:
                high = mid - 1

        # fallback to closest under target
        return best_text, best_len

    # ------------------------- Measurement helpers -------------------------

    def _clear_cuda(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

    def _bytes_per_elem(self) -> int:
        return torch.tensor([], dtype=self._torch_dtype()).element_size()

    def theoretical_kv_mb_per_token(self) -> float:
        """
        Rough theoretical KV per token:
            slope ≈ 2 (K+V) * num_layers * hidden_size * bytes_per_elem
        divided by MiB
        """
        cfg = self.model.config
        n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", 0)))
        hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
        if not n_layers or not hidden:
            return float("nan")
        b = self._bytes_per_elem()
        return (2.0 * n_layers * hidden * b) / (1024.0 * 1024.0)

    # ------------------------- Core run -------------------------

    @torch.inference_mode()
    def run_config(self, prompt_target_tokens: int, gen_tokens_list: List[int]) -> Dict[str, Any]:
        """
        For a fixed prompt length, run several gen lengths, collect peaks,
        fit slope/intercept, and return per-gen metrics plus slope.
        """
        # Construct user content close to target after template
        user_content, prompt_len = self.build_user_content_for_target_tokens(
            prompt_target_tokens, tolerance=2
        )
        formatted = self._apply_template(user_content)
        enc = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_ids = enc["input_ids"]

        # baseline: with model loaded and inputs on device (count it consistently)
        self._clear_cuda()
        baseline_alloc_mb = mib(torch.cuda.memory_allocated(self.device))

        # Warmup (build kernels, caches)
        _ = self.model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        self._clear_cuda()
        baseline_alloc_mb = mib(torch.cuda.memory_allocated(self.device))  # after warm state

        # Now measure decode peaks for several gen lengths
        points: List[RunPoint] = []
        peaks_by_gen: Dict[int, float] = {}

        for gen_tokens in gen_tokens_list:
            self._clear_cuda()
            # prefill+decode in one go
            _ = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                use_cache=True,
            )
            torch.cuda.synchronize()
            peak_alloc_mb = mib(torch.cuda.max_memory_allocated(self.device))
            total_tokens = int(prompt_len + gen_tokens)
            points.append(RunPoint(total_tokens=total_tokens, peak_alloc_mb=peak_alloc_mb))
            peaks_by_gen[gen_tokens] = peak_alloc_mb

        # Fit peak = a + b * total_tokens  (least squares)
        # b ~ MB per token (KV per token), a ~ baseline + overhead
        import numpy as np
        X = np.array([[1.0, p.total_tokens] for p in points], dtype=float)
        y = np.array([p.peak_alloc_mb for p in points], dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # [a, b]
        intercept_mb, slope_mb_per_tok = float(beta[0]), float(beta[1])

        # device properties
        dev = torch.cuda.get_device_properties(self.model.device)
        total_gpu_memory_mb = mib(dev.total_memory)

        # Build rows per generation length
        rows = []
        for gen_tokens in gen_tokens_list:
            total_tokens = int(prompt_len + gen_tokens)
            peak_mb = peaks_by_gen[gen_tokens]
            kv_total_mb = slope_mb_per_tok * total_tokens  # model-agnostic linear estimate

            utilization = 100.0 * (peak_mb / total_gpu_memory_mb)

            rows.append({
                "model": self.model_name,
                "context_length": prompt_len,       # actual after templating
                "gen_tokens": gen_tokens,
                "total_tokens": total_tokens,
                "baseline_alloc_mb": round(baseline_alloc_mb, 3),
                "peak_alloc_mb": round(peak_mb, 3),
                "kv_cache_alloc_mb": round(kv_total_mb, 3),
                "memory_per_token_alloc_mb": round(slope_mb_per_tok, 6),
                "utilization_peak_percent": round(utilization, 6),
                "total_gpu_memory_mb": round(total_gpu_memory_mb, 2),
            })

        # Also print a brief summary for sanity
        print(
            f"[summary] {self.model_name} | prompt={prompt_len} | "
            f"MB/token(slope)={slope_mb_per_tok:.3f} | theory≈{self.theoretical_kv_mb_per_token():.3f}"
        )
        return {
            "prompt_len": prompt_len,
            "rows": rows,
            "slope_mb_per_token": slope_mb_per_tok,
        }


def main():
    parser = argparse.ArgumentParser(description="Fixed memory benchmark with KV slope estimation")
    parser.add_argument("--model", required=True, help="HF model name/path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--prompt_tokens", default="128,512,1024,2048",
                        help="Comma-separated target prompt lengths")
    parser.add_argument("--gen_tokens", default="64,128,256",
                        help="Comma-separated generation lengths for slope fit")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    prompt_targets = [int(x.strip()) for x in args.prompt_tokens.split(",") if x.strip()]
    gen_tokens_list = [int(x.strip()) for x in args.gen_tokens.split(",") if x.strip()]

    bench = MemoryBench(args.model, device=args.device, dtype=args.dtype)
    bench.load()

    all_rows: List[Dict[str, Any]] = []

    # Baseline after load
    baseline_mb = mib(torch.cuda.memory_allocated(bench.model.device))
    dev = torch.cuda.get_device_properties(bench.model.device)
    print(f"[device] total={mib(dev.total_memory):.1f} MiB | baseline_alloc={baseline_mb:.1f} MiB")

    for tgt in prompt_targets:
        print(f"\n[run] Target prompt tokens ≈ {tgt}")
        out = bench.run_config(tgt, gen_tokens_list)
        all_rows.extend(out["rows"])

    # Write CSV matching your schema
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = [
        "model", "context_length", "gen_tokens", "total_tokens",
        "baseline_alloc_mb", "peak_alloc_mb", "kv_cache_alloc_mb",
        "memory_per_token_alloc_mb", "utilization_peak_percent", "total_gpu_memory_mb"
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\n[done] Saved: {args.output}")
    print("Tip: For the paper, report MB/token (slope) and show that it stays stable across prompts.")


if __name__ == "__main__":
    main()
