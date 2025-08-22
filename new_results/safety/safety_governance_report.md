# Safety & Governance Analysis (Qualitative)

_Generated: 2025-08-19T00:59:33.314490Z_

## Environment

- Python: 3.12.3
- Torch: 2.7.1+cu126
- Transformers: 4.55.2
- Accelerator: NVIDIA H100 NVL

## Models

### GPT-OSS-20B (`gpt-oss-20b`)
- **Organization:** Community/OSS
- **License:** TBD (Unknown)
- **Usage Policy:** TBD
- **Governance:** TBD
- **Release Date:** 2024
- **Safety Features (qualitative):** Documentation claims safety-tuned chat formatting (to-verify)

### Qwen3-32B (`qwen3-32b`)
- **Organization:** Alibaba
- **License:** TBD (Restricted)
- **Usage Policy:** TBD
- **Governance:** Alibaba governance (to-verify)
- **Release Date:** 2024
- **Model Card:** https://huggingface.co/Qwen/Qwen3-32B
- **Safety Features (qualitative):** Safety training (per model card), Content filtering (to-verify)
- **Sources:** [model_card](https://huggingface.co/Qwen/Qwen3-32B)

### Yi-34B (`yi-34b`)
- **Organization:** 01.AI
- **License:** TBD (Restricted)
- **Usage Policy:** TBD
- **Governance:** 01.AI governance (to-verify)
- **Release Date:** 2024
- **Model Card:** https://huggingface.co/01-ai/Yi-34B
- **Safety Features (qualitative):** Safety training (per model card), Content moderation (to-verify)
- **Sources:** [model_card](https://huggingface.co/01-ai/Yi-34B)

## Comparative Notes

- **License class counts:** {'Unknown': 1, 'Restricted': 2}
- **Usage policy link present:** {'gpt-oss-20b': False, 'qwen3-32b': False, 'yi-34b': False}
- **Model card link present:** {'gpt-oss-20b': False, 'qwen3-32b': True, 'yi-34b': True}

## Key Findings

- License classes vary across models; confirm exact terms and allowed use before deployment.
- Not all surveyed models have a verified usage policy link in metadata.
- All entries list qualitative safety features; quantitative safety testing not performed in this study.

## Recommendations

- Include authoritative citations for license terms, usage policies, and governance documents.
- Augment this qualitative review with quantitative safety evaluations (harmlessness/jailbreak suites).
- Track model card updates over time and re-run the analysis when versions change.
- Document any deployment-time filters or guardrails alongside model-intrinsic safety features.

> **Limitations:** Documentation review only; no quantitative safety testing was performed.
