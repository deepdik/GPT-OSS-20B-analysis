#!/usr/bin/env python3
"""
Active Parameter Efficiency (APE) Analysis — per-active-B, exact-context
- Uses performance per *active* billion parameters (B) for fair MoE vs Dense comparison.
- Enforces exact (context_length) alignment across latency/memory/energy tables.
- Normalizes schema differences across unified CSVs and fixes suspicious TPOT rows.
- Outputs tidy columns with units in names.

Inputs (CSV paths are relative; edit below if yours differ):
  new_results/unified_latency_comparison.csv
  new_results/unified_memory_comparison_fixed.csv
  new_results/unified_energy_comparison_fixed.csv
Outputs:
  results/ape_analysis.csv
  results/ape_analysis.json
"""

import json
import csv
import os
import math
from typing import Dict, List, Any, Optional, Union
import pandas as pd

Number = Union[int, float]


class APEAnalyzer:
    def __init__(self):
        # --- Model specifications (edit if needed) ---
        self.model_specs = {
            "gpt-oss-20b": {
                "total_params": 20_000_000_000,   # 20B
                "active_params": 3_600_000_000,   # 3.6B (MoE - active experts only)
                "architecture": "MoE",
                "organization": "Community/OSS",
            },
            "qwen3-32b": {
                "total_params": 32_000_000_000,   # 32B
                "active_params": 32_000_000_000,  # dense
                "architecture": "Dense",
                "organization": "Alibaba",
            },
            "yi-34b": {
                "total_params": 34_000_000_000,   # 34B
                "active_params": 34_000_000_000,  # dense
                "architecture": "Dense",
                "organization": "01.AI",
            },
        }

        # name normalization for different files
        self.model_name_map = {
            "gpt-oss-20b": "openai/gpt-oss-20b",
            "qwen3-32b": "Qwen/Qwen3-32B",
            "yi-34b":    "01-ai/Yi-34B",
        }
        # Keep energy consistent with latency/memory names
        self.energy_model_map = {
            "gpt-oss-20b": "openai/gpt-oss-20b",
            "qwen3-32b":   "Qwen/Qwen3-32B",
            "yi-34b":      "01-ai/Yi-34B",
        }

        # CSV locations (change if your paths differ)
        self.latency_csv = "new_results/unified_latency_comparison.csv"
        self.memory_csv  = "new_results/unified_memory_comparison_fixed.csv"
        self.energy_csv  = "new_results/unified_energy_comparison_fixed.csv"

        # Load + pre-process performance data
        self.latency_data = self.load_latency_data()   # includes TPOT fix
        self.memory_data  = self.load_memory_data()
        self.energy_data  = self.load_energy_data()

    # ---------------------- IO helpers ----------------------

    def _read_csv(self, path: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            print(f"Warning: {path} not found")
            return None

    def load_latency_data(self) -> List[Dict[str, Any]]:
        """
        Load latency data and fix suspicious TPOT values:
        If TPOT ≈ 1 / (TTFT_ms/1000), recompute TPOT = gen_tokens / (p50_ms/1000) when possible.
        """
        df = self._read_csv(self.latency_csv)
        if df is None:
            return []

        # Coerce types
        for col in ["TTFT_ms", "TPOT_tokps", "p50_ms", "gen_tokens", "prompt_tokens", "context_length"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        def almost_equal(a: Number, b: Number, rel: float = 0.03) -> bool:
            try:
                if a is None or b is None or math.isnan(float(a)) or math.isnan(float(b)):
                    return False
                return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(b)))
            except Exception:
                return False

        fixed_tpot = []
        for _, row in df.iterrows():
            tpot = row.get("TPOT_tokps")
            ttft = row.get("TTFT_ms")
            if pd.notnull(tpot) and pd.notnull(ttft) and tpot > 0 and ttft > 0:
                approx_from_ttft = 1000.0 / float(ttft)
                if almost_equal(tpot, approx_from_ttft):
                    p50_ms = row.get("p50_ms")
                    gen    = row.get("gen_tokens")
                    if pd.notnull(p50_ms) and pd.notnull(gen) and p50_ms > 0 and gen > 0:
                        tpot = float(gen) / (float(p50_ms) / 1000.0)
            fixed_tpot.append(tpot)

        if "TPOT_tokps" in df.columns:
            df["TPOT_tokps"] = fixed_tpot

        return df.to_dict("records")

    def load_memory_data(self) -> List[Dict[str, Any]]:
        df = self._read_csv(self.memory_csv)
        if df is None:
            return []
        for col in ["context_length", "peak_alloc_mb", "peak_memory_mb", "prompt_tokens"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.to_dict("records")

    def load_energy_data(self) -> List[Dict[str, Any]]:
        df = self._read_csv(self.energy_csv)
        if df is None:
            return []
        for col in [
            "context_length", "prompt_tokens",
            "energy_per_1k_tokens_joules", "energy_per_1k_joules",
            "tokens_per_watt",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.to_dict("records")

    # ---------------------- utilities ----------------------

    def _context_column(self, data: List[Dict[str, Any]]) -> Optional[str]:
        if not data:
            return None
        sample = data[0]
        if "context_length" in sample:
            return "context_length"
        if "prompt_tokens" in sample:
            return "prompt_tokens"
        return None

    def _available_contexts(self, data: List[Dict[str, Any]]) -> List[int]:
        col = self._context_column(data)
        if not col:
            return []
        vals: List[int] = []
        for d in data:
            try:
                c = int(round(float(d.get(col))))
                vals.append(c)
            except Exception:
                pass
        return sorted(list({v for v in vals}))

    def _row_for_context(self, ctx: int, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        col = self._context_column(data)
        if not col:
            return None
        for d in data:
            try:
                c = int(round(float(d.get(col))))
                if c == int(ctx):
                    return d
            except Exception:
                continue
        return None

    @staticmethod
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return default
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _safe_div(n: float, d: float) -> float:
        try:
            return n / d if d else 0.0
        except Exception:
            return 0.0

    # ---------------------- APE core ----------------------

    def calculate_ape_metrics(self) -> List[Dict[str, Any]]:
        ape_results: List[Dict[str, Any]] = []

        print(f"Latency data points: {len(self.latency_data)}")
        print(f"Memory data points:  {len(self.memory_data)}")
        print(f"Energy data points:  {len(self.energy_data)}")

        # Target contexts you care about; we'll enforce exact matches across all three
        target_contexts = [128, 512, 1024, 2048]

        for model_id, specs in self.model_specs.items():
            print(f"\nCalculating APE for {model_id}…")
            name_latency_mem = self.model_name_map[model_id]
            name_energy      = self.energy_model_map[model_id]

            # Filter by model
            L = [d for d in self.latency_data if str(d.get("model", "")).strip() == name_latency_mem]
            M = [d for d in self.memory_data  if str(d.get("model", "")).strip() == name_latency_mem]
            E = [d for d in self.energy_data  if str(d.get("model", "")).strip() == name_energy]

            print(f"  {name_latency_mem}: L={len(L)}  M={len(M)}  E={len(E)}")

            if not (L and M and E):
                print("  Missing one or more datasets; skipping model.")
                continue

            # Exact context alignment
            L_ctx = set(self._available_contexts(L))
            M_ctx = set(self._available_contexts(M))
            E_ctx = set(self._available_contexts(E))
            exact_ctxs = sorted(list((L_ctx & M_ctx & E_ctx) & set(target_contexts)))

            if not exact_ctxs:
                print("  No exact context intersection with targets; skipping model.")
                continue

            # Precompute constants
            active_params_B = specs["active_params"] / 1e9
            total_params_B  = specs["total_params"]  / 1e9

            for ctx in exact_ctxs:
                lrow = self._row_for_context(ctx, L)
                mrow = self._row_for_context(ctx, M)
                erow = self._row_for_context(ctx, E)
                if not all([lrow, mrow, erow]):
                    continue

                # Latency
                ttft_ms = self._to_float(lrow.get("TTFT_ms"), default=0.0)
                tpot    = self._to_float(lrow.get("TPOT_tokps"), default=0.0)

                # Memory (prefer peak_alloc_mb)
                peak_mb = self._to_float(mrow.get("peak_alloc_mb", mrow.get("peak_memory_mb", 0.0)), default=0.0)
                peak_gb = peak_mb / 1024.0

                # Energy
                e1k = erow.get("energy_per_1k_tokens_joules", erow.get("energy_per_1k_joules"))
                energy_per_1k = self._to_float(e1k, default=0.0)
                tokens_per_watt = self._to_float(erow.get("tokens_per_watt"), default=0.0)

                # --- Per-active-B metrics (core APE) ---
                throughput_per_activeB_tokps_per_B = self._safe_div(tpot, active_params_B)
                inv_ttft_tokps = self._safe_div(1000.0, ttft_ms)  # ≈ first-token tok/s
                inv_ttft_per_activeB_tokps_per_B = self._safe_div(inv_ttft_tokps, active_params_B)
                tokens_per_watt_per_activeB = self._safe_div(tokens_per_watt, active_params_B)

                # Memory-normalized performance
                tokens_per_GB_tokps_per_GB = self._safe_div(tpot, peak_gb) if peak_gb > 0 else 0.0

                ape_results.append({
                    # Identification
                    "model_id": model_id,
                    "model_name": name_latency_mem,
                    "architecture": specs["architecture"],
                    "organization": specs["organization"],
                    "context_length": int(ctx),

                    # Params
                    "total_params_B": total_params_B,
                    "active_params_B": active_params_B,

                    # Raw performance (for reference)
                    "ttft_ms": ttft_ms,
                    "tpot_tokps": tpot,
                    "peak_memory_GB": peak_gb,
                    "energy_per_1k_joules": energy_per_1k,
                    "tokens_per_watt": tokens_per_watt,

                    # APE metrics (per-active-B)
                    "throughput_per_activeB_tokps_per_B": throughput_per_activeB_tokps_per_B,
                    "inv_ttft_per_activeB_tokps_per_B":   inv_ttft_per_activeB_tokps_per_B,
                    "tokens_per_watt_per_activeB":        tokens_per_watt_per_activeB,

                    # Memory-normalized performance
                    "tokens_per_GB_tokps_per_GB":         tokens_per_GB_tokps_per_GB,
                })

        return ape_results

    # ---------------------- output ----------------------

    def save_results(self, ape_results: List[Dict[str, Any]]):
        os.makedirs("results", exist_ok=True)

        # CSV
        output_csv = "results/ape_analysis.csv"
        if ape_results:
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(ape_results[0].keys()))
                writer.writeheader()
                writer.writerows(ape_results)
            print(f"APE analysis saved to {output_csv}")

        # JSON
        output_json = "results/ape_analysis.json"
        with open(output_json, "w") as f:
            json.dump({"ape_results": ape_results, "model_specs": self.model_specs}, f, indent=2)
        print(f"APE analysis saved to {output_json}")

    def print_summary(self, ape_results: List[Dict[str, Any]]):
        if not ape_results:
            print("No APE results to summarize.")
            return

        print("\n" + "=" * 120)
        print("ACTIVE PARAMETER EFFICIENCY (APE) — per-active-B summary")
        print("=" * 120)

        # Model specs table
        print("\nMODEL SPECS:")
        print("-" * 80)
        print(f"{'Model':<15} {'Arch':<10} {'Total(B)':<10} {'Active(B)':<10}")
        print("-" * 80)
        for mid, sp in self.model_specs.items():
            print(f"{mid:<15} {sp['architecture']:<10} {sp['total_params']/1e9:<10.1f} {sp['active_params']/1e9:<10.1f}")

        # Largest-context snapshot
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in ape_results:
            groups.setdefault(r["model_id"], []).append(r)

        print("\nAPE @ Largest Common Context:")
        print("-" * 120)
        print(f"{'Model':<15} {'Ctx':<6} {'TPOT/activeB':<14} {'1/TTFT/activeB':<16} {'TPW/activeB':<12} {'TPOT/GB':<12}")
        print("-" * 120)
        for mid in ["gpt-oss-20b", "qwen3-32b", "yi-34b"]:
            if mid not in groups:
                continue
            best = max(groups[mid], key=lambda x: x["context_length"])
            print(f"{mid:<15} {best['context_length']:<6} "
                  f"{best['throughput_per_activeB_tokps_per_B']:<14.3f} "
                  f"{best['inv_ttft_per_activeB_tokps_per_B']:<16.3f} "
                  f"{best['tokens_per_watt_per_activeB']:<12.3f} "
                  f"{best['tokens_per_GB_tokps_per_GB']:<12.3f}")

        print(f"\nTotal APE rows: {len(ape_results)}")
        print("=" * 120)


def main():
    print("Starting APE (per-active-B, exact-context)…")
    analyzer = APEAnalyzer()
    results = analyzer.calculate_ape_metrics()
    analyzer.save_results(results)
    analyzer.print_summary(results)
    return results


if __name__ == "__main__":
    main()
