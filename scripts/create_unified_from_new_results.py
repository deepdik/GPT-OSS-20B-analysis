#!/usr/bin/env python3
"""
Create unified comparison files from new_results folder data
"""

import pandas as pd
import os
from typing import List, Dict, Any

def create_unified_energy_comparison():
    """Create unified energy comparison from new_results energy files"""
    energy_files = [
        "new_results/energy_gptoss20b_fixed.csv",
        "new_results/energy_qwen3-32b_fixed.csv", 
        "new_results/energy_yi-34b_fixed.csv"
    ]
    
    all_data = []
    for file_path in energy_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add model identifier
            if "gptoss20b" in file_path:
                df['model_short'] = 'gpt-oss-20b'
            elif "qwen3-32b" in file_path:
                df['model_short'] = 'qwen3-32b'
            elif "yi-34b" in file_path:
                df['model_short'] = 'yi-34b'
            
            all_data.append(df)
    
    if all_data:
        unified_df = pd.concat(all_data, ignore_index=True)
        # Select and rename key columns
        result_df = unified_df[[
            'model_short', 'model', 'context_length', 'gen_tokens',
            'ttft_ms', 'ttft_joules', 'decode_tokens_per_second', 'total_tokens_per_second',
            'avg_gpu_power_watts', 'baseline_gpu_power_watts',
            'avg_gpu_utilization_percent', 'avg_memory_utilization_percent',
            'energy_per_decoded_token_joules', 'energy_per_1k_decoded_tokens_joules',
            'energy_per_total_token_joules'
        ]].copy()
        
        # Add computed metrics
        result_df['tokens_per_watt'] = result_df['decode_tokens_per_second'] / result_df['avg_gpu_power_watts']
        result_df['energy_per_1k_joules'] = result_df['energy_per_1k_decoded_tokens_joules']
        result_df['TPOT_tokps'] = result_df['decode_tokens_per_second']
        
        os.makedirs("results", exist_ok=True)
        result_df.to_csv("results/unified_energy_comparison_fixed.csv", index=False)
        print(f"Created unified energy comparison with {len(result_df)} rows")
        return result_df
    return None

def create_unified_latency_comparison():
    """Create unified latency comparison from new_results latency files"""
    latency_files = [
        "new_results/latency_gptoss20b_universal.csv",
        "new_results/latency_qwen3-32b_universal.csv",
        "new_results/latency_yi-34b_universal.csv"
    ]
    
    all_data = []
    for file_path in latency_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add model identifier
            if "gptoss20b" in file_path:
                df['model_short'] = 'gpt-oss-20b'
            elif "qwen3-32b" in file_path:
                df['model_short'] = 'qwen3-32b'
            elif "yi-34b" in file_path:
                df['model_short'] = 'yi-34b'
            
            all_data.append(df)
    
    if all_data:
        unified_df = pd.concat(all_data, ignore_index=True)
        # Select and rename key columns
        result_df = unified_df[[
            'model_short', 'model', 'model_type', 'prompt_tokens', 'gen_tokens',
            'TTFT_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'TPOT_tokps'
        ]].copy()
        
        # Rename for consistency
        result_df['context_length'] = result_df['prompt_tokens']
        result_df['ttft_ms'] = result_df['TTFT_ms']
        
        os.makedirs("results", exist_ok=True)
        result_df.to_csv("results/unified_latency_comparison.csv", index=False)
        print(f"Created unified latency comparison with {len(result_df)} rows")
        return result_df
    return None

def create_unified_memory_comparison():
    """Create unified memory comparison from new_results memory files"""
    memory_files = [
        "new_results/memory_gptoss20b_v2.csv",
        "new_results/memory_qwen3-32b_v2.csv",
        "new_results/memory_yi-34b_v2.csv"
    ]
    
    all_data = []
    for file_path in memory_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add model identifier
            if "gptoss20b" in file_path:
                df['model_short'] = 'gpt-oss-20b'
            elif "qwen3-32b" in file_path:
                df['model_short'] = 'qwen3-32b'
            elif "yi-34b" in file_path:
                df['model_short'] = 'yi-34b'
            
            all_data.append(df)
    
    if all_data:
        unified_df = pd.concat(all_data, ignore_index=True)
        # Select and rename key columns
        result_df = unified_df[[
            'model_short', 'model', 'device_name', 'context_length', 'gen_tokens', 'total_tokens',
            'baseline_alloc_mb', 'baseline_reserved_mb',
            'after_tokenize_alloc_mb', 'after_tokenize_reserved_mb',
            'after_generate_alloc_mb', 'after_generate_reserved_mb',
            'peak_alloc_mb', 'peak_reserved_mb',
            'tokenize_delta_alloc_mb', 'generate_delta_alloc_mb', 'total_delta_alloc_mb',
            'kv_cache_alloc_mb', 'kv_cache_reserved_mb',
            'memory_per_token_alloc_mb', 'memory_per_token_reserved_mb',
            'util_after_percent_alloc', 'util_peak_percent_alloc', 'util_peak_percent_reserved',
            'total_gpu_memory_mb'
        ]].copy()
        
        # Add computed metrics
        result_df['peak_memory_mb'] = result_df['peak_alloc_mb']
        result_df['memory_efficiency'] = result_df['total_tokens'] / result_df['peak_alloc_mb']
        
        os.makedirs("results", exist_ok=True)
        result_df.to_csv("results/unified_memory_comparison_fixed.csv", index=False)
        print(f"Created unified memory comparison with {len(result_df)} rows")
        return result_df
    return None

def main():
    print("Creating unified comparison files from new_results...")
    
    # Create unified files
    energy_df = create_unified_energy_comparison()
    latency_df = create_unified_latency_comparison()
    memory_df = create_unified_memory_comparison()
    
    print("\nSummary:")
    if energy_df is not None:
        print(f"Energy: {len(energy_df)} rows, {energy_df['model_short'].nunique()} models")
    if latency_df is not None:
        print(f"Latency: {len(latency_df)} rows, {latency_df['model_short'].nunique()} models")
    if memory_df is not None:
        print(f"Memory: {len(memory_df)} rows, {memory_df['model_short'].nunique()} models")
    
    print("\nFiles created in results/ directory:")
    print("- unified_energy_comparison_fixed.csv")
    print("- unified_latency_comparison.csv")
    print("- unified_memory_comparison_fixed.csv")

if __name__ == "__main__":
    main() 