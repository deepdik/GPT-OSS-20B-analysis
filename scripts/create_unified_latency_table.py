#!/usr/bin/env python3
"""
Create Unified Latency Comparison Table
Combines all latency results into a consistent format for the research paper.
"""

import csv
import json
import os
from typing import Dict, List, Any

def load_latency_data(file_path: str) -> List[Dict[str, Any]]:
    """Load latency data from CSV file"""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return []
    
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def create_unified_table():
    """Create unified latency comparison table"""
    
    # Load all latency data (use complete GPT-OSS-20B data)
    gpt_data = load_latency_data("results/latency_gpt-oss-20b_complete.csv")
    qwen_data = load_latency_data("results/latency_qwen3-32b.csv")
    yi_data = load_latency_data("results/latency_yi-34b.csv")
    
    # Model type mapping
    model_types = {
        "openai/gpt-oss-20b": "chat_template",
        "Qwen/Qwen3-32B": "chat_template", 
        "01-ai/Yi-34B": "completion"
    }
    
    # Create unified data
    unified_data = []
    
    # Process GPT-OSS-20B data (now has model_type column)
    for row in gpt_data:
        unified_data.append({
            "model": row["model"],
            "model_type": row["model_type"],  # Use the column from complete data
            "prompt_tokens": int(row["prompt_tokens"]),
            "gen_tokens": int(row["gen_tokens"]),
            "TTFT_ms": float(row["TTFT_ms"]),
            "p50_ms": float(row["p50_ms"]),
            "p95_ms": float(row["p95_ms"]),
            "p99_ms": float(row["p99_ms"]),
            "TPOT_tokps": float(row["TPOT_tokps"])
        })
    
    # Process Qwen3-32B data
    for row in qwen_data:
        unified_data.append({
            "model": row["model"],
            "model_type": model_types.get(row["model"], "unknown"),
            "prompt_tokens": int(row["prompt_tokens"]),
            "gen_tokens": int(row["gen_tokens"]),
            "TTFT_ms": float(row["TTFT_ms"]),
            "p50_ms": float(row["p50_ms"]),
            "p95_ms": float(row["p95_ms"]),
            "p99_ms": float(row["p99_ms"]),
            "TPOT_tokps": float(row["TPOT_tokps"])
        })
    
    # Process Yi-34B data
    for row in yi_data:
        unified_data.append({
            "model": row["model"],
            "model_type": row["model_type"],
            "prompt_tokens": int(row["prompt_tokens"]),
            "gen_tokens": int(row["gen_tokens"]),
            "TTFT_ms": float(row["TTFT_ms"]),
            "p50_ms": float(row["p50_ms"]),
            "p95_ms": float(row["p95_ms"]),
            "p99_ms": float(row["p99_ms"]),
            "TPOT_tokps": float(row["TPOT_tokps"])
        })
    
    # Save unified CSV
    output_path = "results/unified_latency_comparison.csv"
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'model_type', 'prompt_tokens', 'gen_tokens', 
                     'TTFT_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'TPOT_tokps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unified_data)
    
    print(f"Unified latency comparison saved to {output_path}")
    
    # Create summary table for paper
    create_summary_table(unified_data)
    
    return unified_data

def create_summary_table(data: List[Dict[str, Any]]):
    """Create summary table for the research paper"""
    
    # Filter for 128→64 tokens (standard comparison)
    standard_comparison = [row for row in data if row["prompt_tokens"] == 128 and row["gen_tokens"] == 64]
    
    print("\n" + "="*80)
    print("UNIFIED LATENCY COMPARISON (128→64 tokens)")
    print("="*80)
    print(f"{'Model':<20} {'Type':<15} {'TTFT(ms)':<10} {'P50(ms)':<10} {'TPOT(tok/s)':<12} {'Architecture':<12}")
    print("-"*80)
    
    for row in standard_comparison:
        model_name = row["model"].split("/")[-1]  # Extract short name
        architecture = "MoE" if "gpt-oss" in row["model"] else "Dense"
        print(f"{model_name:<20} {row['model_type']:<15} {row['TTFT_ms']:<10.1f} "
              f"{row['p50_ms']:<10.1f} {row['TPOT_tokps']:<12.1f} {architecture:<12}")
    
    # Save summary to JSON for easy access
    summary_data = {
        "standard_comparison": standard_comparison,
        "all_data": data
    }
    
    with open("results/latency_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary data saved to results/latency_summary.json")

def main():
    print("Creating unified latency comparison table...")
    unified_data = create_unified_table()
    print(f"Total data points: {len(unified_data)}")
    print("Done!")

if __name__ == "__main__":
    main() 