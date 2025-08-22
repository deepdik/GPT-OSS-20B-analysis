# GPT-OSS-20B: A Comprehensive Deployment-Centric Analysis of OpenAI's Open-Weight Mixture of Experts Model

This repository contains a comprehensive deployment-centric analysis of `gpt-oss-20b` on a single NVIDIA H100 GPU, with fair comparisons to 20–40B open-weight peers. It is designed for turnkey reproduction and arXiv-readiness (cs.CL / cs.LG).

## Contributions
- (i) Deployment-centric survey of `gpt-oss-20b` vs 20–40B open-weights
- (ii) Latency/throughput metrics: TTFT, TPOT, p50/p95/p99 under vLLM and Transformers
- (iii) Memory & KV scaling up to long contexts
- (iv) Active-Parameter Efficiency (APE) vs dense peers
- (v) Small safety/governance snapshot

## Repository structure
```
env/          # environment setup and lock files
scripts/      # benchmarking scripts and analysis tools
data/         # small prompt/task stubs, curated safety prompts
new_results/  # working area (raw per-run outputs + unified CSVs)
  ├─ latency/               # per-model latency CSVs
  ├─ energy/                # per-model energy CSVs
  ├─ memeory/               # per-model memory CSVs (name preserved as-is)
  ├─ mmlu-gsm/              # accuracy outputs from lm-eval
  ├─ ablations/             # ablation study outputs
  ├─ safety/                # safety/governance reports
  ├─ ape/                   # APE outputs (optional mirror)
  ├─ unified_latency_comparison.csv
  ├─ unified_memory_comparison_fixed.csv
  └─ unified_energy_comparison_fixed.csv
figs/         # generated plots for latency/KV/APE
paper/        # LaTeX for arXiv (assembled at the end)
```

## Quickstart
```bash
# create & activate venv (Linux)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# install dependencies
pip install -r requirements.txt
```

## Requirements
- NVIDIA GPU with CUDA 12.1+ (tested on H100 80GB)
- Python 3.10–3.12
- PyTorch CUDA wheels (installed above via `--extra-index-url`)

Recommended package set (versions used during our study):

```text
torch (CUDA 12.1 wheels), transformers>=4.43, accelerate>=0.33, tokenizers>=0.19,
vllm>=0.5, lm-eval==0.4.5, pandas>=2.0, matplotlib>=3.8, psutil>=5.9, pynvml>=11.5
```

Always activate the virtual environment before running Python commands:

```bash
source .venv/bin/activate
```

## Models
- gpt-oss-20b: `openai/gpt-oss-20b` (Harmony-format chat, Apache-2.0 + usage policy)
- Comparators: Qwen3-32B, Yi-34B (and optional Llama-3.1-70B 4-bit)

## Planned evaluations
- Accuracy via `lm-evaluation-harness` on MMLU, GSM8K, BBH subset, GPQA/HumanEval (subset)
- Serving metrics via vLLM: TTFT, TPOT, p50/p95/p99 across concurrency and prompt/gen lengths
- Memory & KV scaling: 2k/8k/16k/32k contexts
- Energy snapshot: average W and Wh/1k tokens
- APE analysis: accuracy and latency per active-billion parameters
- Safety snapshot: brief harmlessness/jailbreak set

## Reproducing the study (end-to-end)

Below commands regenerate the core results for the three models on a single GPU. Outputs are written under `new_results/`.

Models used:
- GPT-OSS-20B: `openai/gpt-oss-20b`
- Qwen3-32B: `Qwen/Qwen3-32B`
- Yi-34B: `01-ai/Yi-34B`

### 1) Latency (TTFT, p50/p95/p99, TPOT)
Run for each model (adjust `--prompt_tokens`, `--gen_tokens`, `--num_requests` if desired):

```bash
# GPT-OSS-20B
python scripts/benchmark_latency_universal.py \
  --model openai/gpt-oss-20b \
  --output new_results/latency_gptoss20b_universal.csv \
  --prompt_tokens 128,512,1024,2048 \
  --gen_tokens 64,128,256 \
  --num_requests 10

# Qwen3-32B
python scripts/benchmark_latency_universal.py \
  --model Qwen/Qwen3-32B \
  --output new_results/latency_qwen3-32b_universal.csv \
  --prompt_tokens 128,512,1024,2048 \
  --gen_tokens 64,128,256 \
  --num_requests 10

# Yi-34B
python scripts/benchmark_latency_universal.py \
  --model 01-ai/Yi-34B \
  --output new_results/latency_yi-34b_universal.csv \
  --prompt_tokens 128,512,1024,2048 \
  --gen_tokens 64,128,256 \
  --num_requests 10
```

### 2) Memory (VRAM & KV-cache scaling)
Exact post-template context control; peak-based KV estimation. Run for each model:

```bash
# GPT-OSS-20B
python scripts/memory_analysis_h100_fixed.py \
  --model openai/gpt-oss-20b \
  --output new_results/memory_gptoss20b_v2.csv \
  --context_lengths 128,512,1024,2048 \
  --gen_tokens 64

# Qwen3-32B
python scripts/memory_analysis_h100_fixed.py \
  --model Qwen/Qwen3-32B \
  --output new_results/memory_qwen3-32b_v2.csv \
  --context_lengths 128,512,1024,2048 \
  --gen_tokens 64

# Yi-34B
python scripts/memory_analysis_h100_fixed.py \
  --model 01-ai/Yi-34B \
  --output new_results/memory_yi-34b_v2.csv \
  --context_lengths 128,512,1024,2048 \
  --gen_tokens 64
```

### 3) Energy (decode-centric throughput and efficiency)
Uses `nvidia-smi` polling. Recommended: `--power_poll_hz 10` (10 Hz). Run for each model:

```bash
# GPT-OSS-20B
python scripts/energy_analysis_simple.py \
  --model openai/gpt-oss-20b \
  --output new_results/energy_gptoss20b_fixed.csv \
  --context_lengths 128,512,1024,2048 \
  --gen_tokens 64 \
  --num_runs 5 \
  --device cuda:0 \
  --power_poll_hz 10

# Qwen3-32B
python scripts/energy_analysis_simple.py \
  --model Qwen/Qwen3-32B \
  --output new_results/energy_qwen3-32b_fixed.csv \
  --context_lengths 128,512,1024,2048 \
  --gen_tokens 64 \
  --num_runs 5 \
  --device cuda:0 \
  --power_poll_hz 10

# Yi-34B
python scripts/energy_analysis_simple.py \
  --model 01-ai/Yi-34B \
  --output new_results/energy_yi-34b_fixed.csv \
  --context_lengths 128,512,1024,2048 \
  --gen_tokens 64 \
  --num_runs 5 \
  --device cuda:0 \
  --power_poll_hz 10
```

### 4) Consolidate to unified CSVs

```bash
python scripts/create_unified_from_new_results.py
# Creates unified CSVs in new_results/: unified_latency_comparison.csv, unified_memory_comparison_fixed.csv, unified_energy_comparison_fixed.csv
```

### 5) APE (Active Parameter Efficiency)

```bash
python scripts/ape_analysis.py
# Writes: new_results/ape/ape_analysis.csv and new_results/ape/ape_analysis.json
```

### 6) Ablation studies (optional)

```bash
python scripts/ablation_studies.py \
  --model openai/gpt-oss-20b \
  --output_dir new_results/ablations \
  --tests all \
  --context_lengths 512,1024,2048 \
  --num_runs 5 \
  --max_new_tokens 64
```

### 7) Safety & Governance (qualitative)

```bash
python scripts/safety_governance_analysis.py --output_dir new_results/safety
```

### 8) Accuracy (optional; requires lm-eval datasets)

Example command (adjust for your environment):

```bash
lm_eval \
  --model hf \
  --model_args pretrained=openai/gpt-oss-20b,trust_remote_code=True \
  --tasks mmlu,gsm8k_cot \
  --apply_chat_template \
  --batch_size 1 \
  --output_path new_results/mmlu-gsm/gptoss20b.json
```

## Reproducibility notes
- All Python commands assume an active virtual environment (`source .venv/bin/activate`).
- Seeds are set in the scripts where applicable to reduce jitter.
- We record executed benchmarking commands under `results/model_specs/` when applicable. You can also log your exact CLI by saving it to a text file next to the outputs.

## References
- OpenAI: [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/)
- Model card (gpt-oss-120b & gpt-oss-20b): https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf
- OpenAI Platform docs (21B with ~3.6B active): https://platform.openai.com/docs/models/gpt-oss-20b
- Harmony format: https://github.com/openai/harmony
- vLLM day-0 guide: https://cookbook.openai.com/articles/gpt-oss/run-vllm
- Transformers guide: https://cookbook.openai.com/articles/gpt-oss/run-transformers
- Qwen3-32B: https://huggingface.co/Qwen/Qwen3-32B
- Yi-34B: https://huggingface.co/01-ai/Yi-34B

## License
Code in this repository is licensed under the Apache License, Version 2.0. See `LICENSE`.
