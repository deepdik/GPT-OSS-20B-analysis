#!/usr/bin/env python3
"""
Create unified energy comparison from fixed energy analysis results
"""

import csv
import json
import os
from typing import Dict, List, Any

def load_energy_data(file_path: str) -> List[Dict[str, Any]]:
    """Load energy data from CSV file"""
    if not os.path.exists(file_path):
        return []
    
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def create_unified_energy_table():
    """Create unified energy comparison table"""
    print("Creating unified energy comparison table...")
    
    # Load energy data from fixed files
    gpt_data = load_energy_data("results/energy_gpt-oss-20b_fixed.csv")
    qwen_data = load_energy_data("results/energy_qwen3-32b_fixed.csv")
    yi_data = load_energy_data("results/energy_yi-34b_fixed.csv")
    
    unified_data = []
    
    # Process GPT-OSS-20B data
    for row in gpt_data:
        unified_data.append({
            "model": "gpt-oss-20b",
            "architecture": "MoE",
            "context_length": int(row["context_length"]),
            "power_watts": float(row["avg_gpu_power_watts"]),
            "throughput_tokps": float(row["tokens_per_second"]),
            "energy_per_1k_joules": float(row["energy_per_1k_tokens_joules"]),
            "gpu_utilization_percent": float(row["avg_gpu_utilization_percent"]),
            "tokens_per_watt": float(row["tokens_per_watt"])
        })
    
    # Process Qwen3-32B data
    for row in qwen_data:
        unified_data.append({
            "model": "Qwen3-32B",
            "architecture": "Dense",
            "context_length": int(row["context_length"]),
            "power_watts": float(row["avg_gpu_power_watts"]),
            "throughput_tokps": float(row["tokens_per_second"]),
            "energy_per_1k_joules": float(row["energy_per_1k_tokens_joules"]),
            "gpu_utilization_percent": float(row["avg_gpu_utilization_percent"]),
            "tokens_per_watt": float(row["tokens_per_watt"])
        })
    
    # Process Yi-34B data
    for row in yi_data:
        unified_data.append({
            "model": "Yi-34B",
            "architecture": "Dense",
            "context_length": int(row["context_length"]),
            "power_watts": float(row["avg_gpu_power_watts"]),
            "throughput_tokps": float(row["tokens_per_second"]),
            "energy_per_1k_joules": float(row["energy_per_1k_tokens_joules"]),
            "gpu_utilization_percent": float(row["avg_gpu_utilization_percent"]),
            "tokens_per_watt": float(row["tokens_per_watt"])
        })
    
    # Save unified CSV
    output_path = "results/unified_energy_comparison_fixed.csv"
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'architecture', 'context_length', 'power_watts', 
                     'throughput_tokps', 'energy_per_1k_joules', 'gpu_utilization_percent', 'tokens_per_watt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unified_data)
    
    print(f"Unified energy comparison saved to {output_path}")
    
    # Create summary data
    model_summaries = {}
    for row in unified_data:
        model = row["model"]
        ctx_length = row["context_length"]
        if model not in model_summaries:
            model_summaries[model] = {}
        model_summaries[model][ctx_length] = row
    
    # Save summary JSON
    summary_path = "results/energy_summary_fixed.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "model_summaries": model_summaries,
            "all_data": unified_data
        }, f, indent=2)
    
    print(f"Summary data saved to {summary_path}")
    
    # Print summary table
    print("\n" + "="*120)
    print("UNIFIED ENERGY COMPARISON SUMMARY (FIXED CONTEXT LENGTHS)")
    print("="*120)
    print(f"{'Model':<15} {'Architecture':<12} {'Context':<8} {'Power(W)':<10} {'Throughput':<12} {'Energy/1k(J)':<12} {'GPU Util':<10} {'Tokens/W':<10}")
    print("-"*120)
    
    for model_name in ["gpt-oss-20b", "Qwen3-32B", "Yi-34B"]:
        if model_name in model_summaries:
            architecture = "MoE" if model_name == "gpt-oss-20b" else "Dense"
            for ctx_length in [128, 512, 1024, 2048]:
                if ctx_length in model_summaries[model_name]:
                    data = model_summaries[model_name][ctx_length]
                    print(f"{model_name:<15} {architecture:<12} {ctx_length:<8} "
                          f"{data['power_watts']:<10.1f} "
                          f"{data['throughput_tokps']:<12.1f} "
                          f"{data['energy_per_1k_joules']:<12.1f} "
                          f"{data['gpu_utilization_percent']:<10.1f}% "
                          f"{data['tokens_per_watt']:<10.2f}")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY ENERGY FINDINGS")
    print("="*80)
    
    # Compare 2048 context results
    gpt_2048 = model_summaries["gpt-oss-20b"][2048]
    qwen_2048 = model_summaries["Qwen3-32B"][2048]
    yi_2048 = model_summaries["Yi-34B"][2048]
    
    print(f"GPT-OSS-20B (MoE) - 2048 context:")
    print(f"  Power: {gpt_2048['power_watts']:.1f}W, Throughput: {gpt_2048['throughput_tokps']:.1f} tok/s")
    print(f"  Energy/1k tokens: {gpt_2048['energy_per_1k_joules']:.1f}J, Efficiency: {gpt_2048['tokens_per_watt']:.2f} tok/W")
    
    print(f"\nQwen3-32B (Dense) - 2048 context:")
    print(f"  Power: {qwen_2048['power_watts']:.1f}W, Throughput: {qwen_2048['throughput_tokps']:.1f} tok/s")
    print(f"  Energy/1k tokens: {qwen_2048['energy_per_1k_joules']:.1f}J, Efficiency: {qwen_2048['tokens_per_watt']:.2f} tok/W")
    
    print(f"\nYi-34B (Dense) - 2048 context:")
    print(f"  Power: {yi_2048['power_watts']:.1f}W, Throughput: {yi_2048['throughput_tokps']:.1f} tok/s")
    print(f"  Energy/1k tokens: {yi_2048['energy_per_1k_joules']:.1f}J, Efficiency: {yi_2048['tokens_per_watt']:.2f} tok/W")
    
    # Calculate advantages
    power_advantage = ((qwen_2048['power_watts'] - gpt_2048['power_watts']) / qwen_2048['power_watts']) * 100
    throughput_advantage = ((gpt_2048['throughput_tokps'] - qwen_2048['throughput_tokps']) / qwen_2048['throughput_tokps']) * 100
    energy_advantage = ((qwen_2048['energy_per_1k_joules'] - gpt_2048['energy_per_1k_joules']) / qwen_2048['energy_per_1k_joules']) * 100
    efficiency_advantage = ((gpt_2048['tokens_per_watt'] - qwen_2048['tokens_per_watt']) / qwen_2048['tokens_per_watt']) * 100
    
    print(f"\nGPT-OSS-20B (MoE) Advantages:")
    print(f"  Power Usage: {power_advantage:.1f}% lower than Qwen3-32B")
    print(f"  Throughput: {throughput_advantage:.1f}% higher than Qwen3-32B")
    print(f"  Energy Efficiency: {energy_advantage:.1f}% lower energy per 1k tokens")
    print(f"  Power Efficiency: {efficiency_advantage:.1f}% higher tokens per watt")
    
    print(f"\nTotal data points: {len(unified_data)}")
    print("Done!")

if __name__ == "__main__":
    create_unified_energy_table() 