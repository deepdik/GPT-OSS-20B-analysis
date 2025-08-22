#!/usr/bin/env python3
"""
Energy Analysis (paper-ready, single GPU)

Fixes vs. previous:
- Pins nvidia-smi queries to the same GPU as --device (e.g., cuda:0)
- Adds TTFT energy (J) from a true 1-token generate
- Optional power integration during decode via --power_poll_hz (e.g., 10 Hz)
- Reports decode-centric metrics (J/decoded-token, decoded tok/s)

Note: Power from nvidia-smi is approximate. We run multiple trials and average.
"""

import argparse
import csv
import gc
import os
import re
import subprocess
import threading
import time
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_gpu_index(device: str) -> int:
    """
    Extract GPU index from device spec like 'cuda:0' or 'cuda'.
    Defaults to 0 if unspecified.
    """
    if device is None:
        return 0
    m = re.match(r"cuda:(\d+)", str(device))
    return int(m.group(1)) if m else 0


class PowerSampler(threading.Thread):
    """Background sampler to poll GPU power at fixed frequency (Hz)."""
    def __init__(self, gpu_index: int, poll_hz: float):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.poll_hz = max(0.0, float(poll_hz))
        self.samples: List[Tuple[float, float]] = []
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        if self.poll_hz <= 0:
            return
        period = 1.0 / self.poll_hz
        while not self._stop_evt.is_set():
            t = time.time()
            p = self._read_power()
            self.samples.append((t, p))
            # sleep but allow prompt stop
            self._stop_evt.wait(period)
        # one last sample at stop time
        self.samples.append((time.time(), self._read_power()))

    def _read_power(self) -> float:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_index),
                 "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip().splitlines()[0])
        except Exception:
            pass
        return 0.0

    def integrated_energy_joules(self) -> Tuple[float, float]:
        """
        Trapezoidal integration over (time, power) samples.
        Returns (energy_joules, avg_power_watts).
        """
        s = self.samples
        if len(s) < 2:
            return 0.0, (s[0][1] if s else 0.0)
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(s[:-1], s[1:]):
            dt = max(0.0, t1 - t0)
            energy += 0.5 * (p0 + p1) * dt
        total_t = max(1e-9, s[-1][0] - s[0][0])
        avg_p = energy / total_t
        return energy, avg_p


class EnergyAnalyzer:
    def __init__(self, model_name: str, device: str = "cuda:0",
                 power_poll_hz: float = 0.0):
        self.model_name = model_name
        self.device = device
        self.gpu_index = parse_gpu_index(device)
        self.power_poll_hz = power_poll_hz

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    # ---------- GPU telemetry ----------

    def get_gpu_power(self) -> float:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_index),
                 "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip().splitlines()[0])
        except Exception as e:
            print(f"Warning: Could not get GPU power: {e}")
        return 0.0

    def get_gpu_utilization(self) -> Tuple[float, float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.gpu_index),
                 "--query-gpu=utilization.gpu,utilization.memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                line = result.stdout.strip().splitlines()[0]
                gpu_util, mem_util = map(float, line.split(","))
                return gpu_util, mem_util
        except Exception as e:
            print(f"Warning: Could not get GPU utilization: {e}")
        return 0.0, 0.0

    # ---------- Model setup ----------

    def load_model(self):
        print(f"Loading {self.model_name} on {self.device} (GPU {self.gpu_index})...")
        assert torch.cuda.is_available(), "CUDA not available"

        # Ensure we are on the intended GPU
        torch.cuda.set_device(self.gpu_index)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            # fallback for decoder-only
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.cuda.synchronize()
        print(f"Model loaded on {self.device}: {torch.cuda.get_device_name(self.gpu_index)}")

    # ---------- Prompt building ----------

    def _base_text(self) -> str:
        return (
            "The following is a comprehensive analysis of artificial intelligence and "
            "machine learning technologies. We explore deep learning architectures, "
            "natural language processing, computer vision, and real-world systems."
        )

    def generate_prompt_exact_tokens(self, target_tokens: int) -> str:
        """
        Build a text that tokenizes to (approximately) target_tokens;
        then trim *after* tokenization to match exactly.
        """
        base = self._base_text()
        # repeat text to exceed target
        base_ids = self.tokenizer.encode(base, add_special_tokens=False)
        reps = max(1, (target_tokens // max(1, len(base_ids))) + 1)
        text = (" " + base) * reps
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[:target_tokens]
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    def build_inputs(self, context_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = self.generate_prompt_exact_tokens(context_length)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        return input_ids, attention_mask

    # ---------- Warm-up & TTFT ----------

    def warm_up(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.inference_mode():
            _ = self.model.generate(
                input_ids=input_ids[:, :min(16, input_ids.size(1))],
                attention_mask=attention_mask[:, :min(16, attention_mask.size(1))],
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        torch.cuda.synchronize()

    def measure_ttft_ms_and_joules(self, input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor) -> Tuple[float, float]:
        """True TTFT for exactly 1 new token, plus energy (J) using start/end power."""
        torch.cuda.synchronize()
        p0 = self.get_gpu_power()
        t0 = time.time()
        with torch.inference_mode():
            _ = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False
            )
        torch.cuda.synchronize()
        t1 = time.time()
        p1 = self.get_gpu_power()
        dt = t1 - t0
        ttft_ms = dt * 1000.0
        # simple trapezoid using two samples
        ttft_energy_j = ((p0 + p1) / 2.0) * dt
        return ttft_ms, ttft_energy_j

    # ---------- Measurement loop ----------

    def measure_config(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       gen_tokens: int, num_runs: int, context_length: int) -> Dict[str, Any]:
        print(f"  ctx={context_length} tokens | gen={gen_tokens} | runs={num_runs}")

        # Baseline telemetry
        baseline_power = self.get_gpu_power()
        base_gpu_util, base_mem_util = self.get_gpu_utilization()

        # TTFT (one-time per config)
        self.warm_up(input_ids, attention_mask)
        ttft_ms, ttft_j = self.measure_ttft_ms_and_joules(input_ids, attention_mask)

        gen_times, avg_powers, avg_gpu_utils, avg_mem_utils, new_tok_counts = [], [], [], [], []

        for r in range(num_runs):
            # Optional integrated sampling
            sampler = PowerSampler(self.gpu_index, self.power_poll_hz)
            sampler.start()

            p_start = self.get_gpu_power()
            u_start, m_start = self.get_gpu_utilization()

            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                    output_hidden_states=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            torch.cuda.synchronize()
            t1 = time.time()

            p_end = self.get_gpu_power()
            u_end, m_end = self.get_gpu_utilization()

            # stop sampler and compute energy/power
            sampler.stop()
            sampler.join(timeout=1.0)
            if self.power_poll_hz > 0 and len(sampler.samples) >= 2:
                energy_j, avg_p = sampler.integrated_energy_joules()
            else:
                # fallback: average of start/end over wall time
                energy_j = 0.5 * (p_start + p_end) * (t1 - t0)
                avg_p = 0.5 * (p_start + p_end)

            gen_time = max(1e-9, t1 - t0)
            new_toks = out.sequences.shape[1] - input_ids.shape[1]

            gen_times.append(gen_time)
            avg_powers.append(avg_p)
            avg_gpu_utils.append(0.5 * (u_start + u_end))
            avg_mem_utils.append(0.5 * (m_start + m_end))
            new_tok_counts.append(new_toks)

        # Aggregates
        avg_gen_time = sum(gen_times) / len(gen_times)
        avg_new_toks = sum(new_tok_counts) / len(new_tok_counts)
        decode_tok_per_s = avg_new_toks / avg_gen_time

        avg_power = sum(avg_powers) / len(avg_powers)
        avg_gpu_util = sum(avg_gpu_utils) / len(avg_gpu_utils)
        avg_mem_util = sum(avg_mem_utils) / len(avg_mem_utils)

        # Energy per decoded token (preferred) using avg power & time:
        # (If --power_poll_hz>0, avg_power reflects the integrated average)
        energy_per_decode_j = (avg_power * avg_gen_time) / max(1e-9, avg_new_toks)
        energy_per_1k_decode_j = energy_per_decode_j * 1000.0

        # Also report the "total tokens" view for completeness (optional)
        total_tokens = input_ids.shape[1] + avg_new_toks
        total_tok_per_s = total_tokens / avg_gen_time
        energy_per_total_tok_j = (avg_power * avg_gen_time) / max(1e-9, total_tokens)

        return {
            "model": self.model_name,
            "device": self.device,
            "gpu_index": self.gpu_index,
            "context_length": context_length,
            "gen_tokens": gen_tokens,
            "num_runs": num_runs,

            # TTFT
            "ttft_ms": ttft_ms,
            "ttft_joules": ttft_j,

            # Throughput
            "decode_tokens_per_second": decode_tok_per_s,
            "total_tokens_per_second": total_tok_per_s,

            # Power & util
            "avg_gpu_power_watts": avg_power,
            "baseline_gpu_power_watts": baseline_power,
            "avg_gpu_utilization_percent": avg_gpu_util,
            "avg_memory_utilization_percent": avg_mem_util,
            "baseline_gpu_utilization_percent": base_gpu_util,
            "baseline_memory_utilization_percent": base_mem_util,

            # Energy (decode-centric)
            "energy_per_decoded_token_joules": energy_per_decode_j,
            "energy_per_1k_decoded_tokens_joules": energy_per_1k_decode_j,

            # Energy (total-token view, optional)
            "energy_per_total_token_joules": energy_per_total_tok_j,
        }

    def run(self, context_lengths: List[int], gen_tokens: int, num_runs: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        print(f"Starting energy analysis for {self.model_name}")
        print(f"GPU {self.gpu_index} | ctxs={context_lengths} | gen={gen_tokens} | runs={num_runs} | poll_hz={self.power_poll_hz}")
        for ctx in context_lengths:
            try:
                inp, attn = self.build_inputs(ctx)
                # brief warm-up
                self.warm_up(inp, attn)
                res = self.measure_config(inp, attn, gen_tokens, num_runs, ctx)
                results.append(res)
                print(f"    ctx={ctx:5d} | P~{res['avg_gpu_power_watts']:.1f} W | "
                      f"decode={res['decode_tokens_per_second']:.1f} tok/s | "
                      f"E/1k-decode={res['energy_per_1k_decoded_tokens_joules']:.1f} J | "
                      f"TTFT={res['ttft_ms']:.1f} ms, {res['ttft_joules']:.2f} J")
            except Exception as e:
                print(f"[warn] ctx {ctx}: {e}")
        return results

    @staticmethod
    def save_csv(rows: List[Dict[str, Any]], out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not rows:
            return
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def print_summary(rows: List[Dict[str, Any]]):
        if not rows:
            return
        print("\n" + "=" * 92)
        print("ENERGY SUMMARY (decode-centric)")
        print("=" * 92)
        print(f"{'ctx':>6}  {'P_avg(W)':>9}  {'decode tok/s':>13}  {'E/1k-decode(J)':>16}  {'TTFT(ms)':>9}  {'TTFT(J)':>8}")
        for r in rows:
            print(f"{r['context_length']:6d}  {r['avg_gpu_power_watts']:9.1f}  "
                  f"{r['decode_tokens_per_second']:13.1f}  {r['energy_per_1k_decoded_tokens_joules']:16.1f}  "
                  f"{r['ttft_ms']:9.1f}  {r['ttft_joules']:8.2f}")


def main():
    ap = argparse.ArgumentParser(description="Energy Analysis (paper-ready)")
    ap.add_argument("--model", required=True, help="Model name/path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--context_lengths", default="128,512,1024,2048",
                    help="Comma-separated context lengths")
    ap.add_argument("--gen_tokens", type=int, default=64,
                    help="Decoded tokens per run")
    ap.add_argument("--num_runs", type=int, default=5,
                    help="Repetitions per (ctx,gen)")
    ap.add_argument("--device", default="cuda:0",
                    help="Device, e.g., cuda:0")
    ap.add_argument("--power_poll_hz", type=float, default=0.0,
                    help="If >0, integrates power at this Hz during decode")
    args = ap.parse_args()

    ctxs = [int(x.strip()) for x in args.context_lengths.split(",") if x.strip()]

    # Repro-ish seeds (doesn't affect deterministic kernels)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    analyzer = EnergyAnalyzer(args.model, device=args.device, power_poll_hz=args.power_poll_hz)
    analyzer.load_model()
    rows = analyzer.run(ctxs, args.gen_tokens, args.num_runs)
    analyzer.save_csv(rows, args.output)
    print(f"\nSaved: {args.output}")
    analyzer.print_summary(rows)


if __name__ == "__main__":
    main()
