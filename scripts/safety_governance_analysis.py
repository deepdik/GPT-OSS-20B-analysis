#!/usr/bin/env python3
"""
Safety and Governance Analysis (paper-ready, metadata-driven)

- Normalizes model IDs to match the rest of your pipeline: gpt-oss-20b, qwen3-32b, yi-34b
- Allows supplying a verified metadata JSON (links, license, governance, usage policy)
- Produces JSON, CSV, and Markdown report suitable for paper appendix
- Records environment info and clearly states qualitative-only limitation
"""

import json
import os
import sys
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# --------------------------- Defaults (safe placeholders) ---------------------------

DEFAULT_MODELS: Dict[str, Dict[str, Any]] = {
    # NOTE: Fill these with verified sources via --metadata whenever possible.
    "gpt-oss-20b": {
        "id": "gpt-oss-20b",
        "name": "GPT-OSS-20B",
        "organization": "Community/OSS",                 # keep consistent with APE section
        "license": "TBD",                                 # e.g., "Apache-2.0"
        "license_class": "Unknown",                       # Permissive / Restricted / Research-only / Unknown
        "usage_policy": "TBD",                            # short label
        "safety_features": [
            "Documentation claims safety-tuned chat formatting (to-verify)",
        ],
        "governance": "TBD",
        "release_date": "2024",
        "model_card": "",                                 # URL, to-verify
        "sources": {                                      # add urls for citations
            "license": "",
            "usage_policy": "",
            "model_card": "",
            "governance": ""
        }
    },
    "qwen3-32b": {
        "id": "qwen3-32b",
        "name": "Qwen3-32B",
        "organization": "Alibaba",
        "license": "TBD",            # e.g., "Qwen License"
        "license_class": "Restricted",
        "usage_policy": "TBD",
        "safety_features": [
            "Safety training (per model card)",
            "Content filtering (to-verify)"
        ],
        "governance": "Alibaba governance (to-verify)",
        "release_date": "2024",
        "model_card": "https://huggingface.co/Qwen/Qwen3-32B",
        "sources": {
            "license": "",
            "usage_policy": "",
            "model_card": "https://huggingface.co/Qwen/Qwen3-32B",
            "governance": ""
        }
    },
    "yi-34b": {
        "id": "yi-34b",
        "name": "Yi-34B",
        "organization": "01.AI",
        "license": "TBD",            # e.g., "Yi License"
        "license_class": "Restricted",
        "usage_policy": "TBD",
        "safety_features": [
            "Safety training (per model card)",
            "Content moderation (to-verify)"
        ],
        "governance": "01.AI governance (to-verify)",
        "release_date": "2024",
        "model_card": "https://huggingface.co/01-ai/Yi-34B",
        "sources": {
            "license": "",
            "usage_policy": "",
            "model_card": "https://huggingface.co/01-ai/Yi-34B",
            "governance": ""
        }
    }
}

# --------------------------- Utilities ---------------------------

def load_metadata(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load metadata JSON if provided; otherwise return defaults."""
    if not path:
        return DEFAULT_MODELS
    with open(path, "r") as f:
        data = json.load(f)
    # Expect either a dict keyed by model-id, or a list of records each with 'id'
    if isinstance(data, list):
        meta = {}
        for rec in data:
            mid = rec.get("id")
            if mid:
                meta[mid] = rec
        return meta or DEFAULT_MODELS
    elif isinstance(data, dict):
        return data
    return DEFAULT_MODELS


def normalize_model_record(model_id: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required fields exist, fill placeholders if missing."""
    base = DEFAULT_MODELS[model_id].copy()
    base.update(rec or {})
    base.setdefault("id", model_id)
    base.setdefault("name", base["name"])
    base.setdefault("organization", "TBD")
    base.setdefault("license", "TBD")
    base.setdefault("license_class", "Unknown")
    base.setdefault("usage_policy", "TBD")
    base.setdefault("safety_features", [])
    base.setdefault("governance", "TBD")
    base.setdefault("release_date", "TBD")
    base.setdefault("model_card", "")
    base.setdefault("sources", {"license": "", "usage_policy": "", "model_card": "", "governance": ""})
    # Basic validation
    if not isinstance(base["safety_features"], list):
        base["safety_features"] = [str(base["safety_features"])]
    if not isinstance(base["sources"], dict):
        base["sources"] = {"license": "", "usage_policy": "", "model_card": "", "governance": ""}
    return base


def gather_env_info() -> Dict[str, Any]:
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        import torch  # type: ignore
        torch_ver = getattr(torch, "__version__", "unknown")
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only"
    except Exception:
        torch_ver = "unavailable"
        gpu = "unknown"
    try:
        import transformers  # type: ignore
        tf_ver = getattr(transformers, "__version__", "unknown")
    except Exception:
        tf_ver = "unavailable"
    return {
        "python": py,
        "torch": torch_ver,
        "transformers": tf_ver,
        "accelerator": gpu,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def license_tier_summary(models: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    tiers = {}
    for _, m in models.items():
        t = (m.get("license_class") or "Unknown").strip()
        tiers[t] = tiers.get(t, 0) + 1
    return tiers


def write_csv_overview(models: Dict[str, Dict[str, Any]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = [
        "id", "name", "organization", "license", "license_class", "usage_policy",
        "governance", "release_date", "model_card",
        "sources.license", "sources.usage_policy", "sources.model_card", "sources.governance"
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for mid, m in models.items():
            s = m.get("sources", {})
            w.writerow([
                m.get("id",""), m.get("name",""), m.get("organization",""),
                m.get("license",""), m.get("license_class",""), m.get("usage_policy",""),
                m.get("governance",""), m.get("release_date",""), m.get("model_card",""),
                s.get("license",""), s.get("usage_policy",""), s.get("model_card",""), s.get("governance","")
            ])


# --------------------------- Analyzer ---------------------------

class SafetyGovernanceAnalyzer:
    def __init__(self, models: Dict[str, Dict[str, Any]]):
        # Normalize all model records and keep only the three we compare
        wanted = ["gpt-oss-20b", "qwen3-32b", "yi-34b"]
        self.models: Dict[str, Dict[str, Any]] = {}
        for mid in wanted:
            rec = normalize_model_record(mid, models.get(mid, {}))
            self.models[mid] = rec

    def create_safety_summary(self) -> Dict[str, Any]:
        env = gather_env_info()
        summary = {
            "analysis_date": env["timestamp"],
            "environment": env,
            "models": self.models,
            "comparative_analysis": {},
            "key_findings": [],
            "recommendations": []
        }

        # Comparative analysis (data-driven, no unverified adjectives)
        summary["comparative_analysis"] = {
            "license_classes": license_tier_summary(self.models),
            "has_usage_policy": {mid: bool(m.get("usage_policy") and m.get("usage_policy") != "TBD")
                                 for mid, m in self.models.items()},
            "has_model_card_link": {mid: bool(m.get("model_card")) for mid, m in self.models.items()},
        }

        # Key findings (tempered language)
        kf = []
        # License classes
        kf.append("License classes vary across models; confirm exact terms and allowed use before deployment.")
        # Usage policy coverage
        if all(summary["comparative_analysis"]["has_usage_policy"].values()):
            kf.append("All surveyed models include a documented usage policy (verify links in metadata).")
        else:
            kf.append("Not all surveyed models have a verified usage policy link in metadata.")
        # Safety features (qualitative)
        kf.append("All entries list qualitative safety features; quantitative safety testing not performed in this study.")
        summary["key_findings"] = kf

        # Recommendations (paper-suitable)
        summary["recommendations"] = [
            "Include authoritative citations for license terms, usage policies, and governance documents.",
            "Augment this qualitative review with quantitative safety evaluations (harmlessness/jailbreak suites).",
            "Track model card updates over time and re-run the analysis when versions change.",
            "Document any deployment-time filters or guardrails alongside model-intrinsic safety features."
        ]
        return summary

    def run_safety_evaluation(self) -> Dict[str, Any]:
        # Qualitative placeholder (explicitly marked)
        return {
            "evaluation_method": "Documentation review (qualitative only)",
            "note": "No quantitative safety probing was performed in this analysis.",
            "suggested_next_steps": [
                "Run a curated harmlessness/jailbreak prompt suite.",
                "Measure refusal rates, toxicity scores, and policy compliance.",
                "Evaluate effectiveness of moderation filters end-to-end."
            ],
            "models": {
                mid: {
                    "safety_score": "N/A (qualitative)",
                    "strengths": m.get("safety_features", [])[:3],
                    "limitations": [
                        "Quantitative safety effectiveness not measured",
                        "Based on public documentation; may be incomplete"
                    ]
                }
                for mid, m in self.models.items()
            }
        }

    def write_reports(self, summary: Dict[str, Any], evaluation: Dict[str, Any], outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        # JSON outputs
        with open(os.path.join(outdir, "safety_governance_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(outdir, "safety_evaluation.json"), "w") as f:
            json.dump(evaluation, f, indent=2)
        # CSV overview for appendix
        write_csv_overview(self.models, os.path.join(outdir, "safety_governance_overview.csv"))
        # Markdown report (paper appendix friendly)
        md = os.path.join(outdir, "safety_governance_report.md")
        with open(md, "w") as f:
            f.write("# Safety & Governance Analysis (Qualitative)\n\n")
            f.write(f"_Generated: {summary['analysis_date']}_\n\n")
            f.write("## Environment\n\n")
            env = summary["environment"]
            f.write(f"- Python: {env['python']}\n- Torch: {env['torch']}\n- Transformers: {env['transformers']}\n- Accelerator: {env['accelerator']}\n\n")

            f.write("## Models\n\n")
            for mid, m in self.models.items():
                f.write(f"### {m['name']} (`{mid}`)\n")
                f.write(f"- **Organization:** {m['organization']}\n")
                f.write(f"- **License:** {m['license']} ({m['license_class']})\n")
                f.write(f"- **Usage Policy:** {m['usage_policy']}\n")
                f.write(f"- **Governance:** {m['governance']}\n")
                f.write(f"- **Release Date:** {m['release_date']}\n")
                if m.get("model_card"):
                    f.write(f"- **Model Card:** {m['model_card']}\n")
                if m.get("safety_features"):
                    f.write(f"- **Safety Features (qualitative):** {', '.join(m['safety_features'])}\n")
                # Footnote-style sources
                src = m.get("sources", {})
                foots = []
                for label, url in src.items():
                    if url:
                        foots.append(f"[{label}]({url})")
                if foots:
                    f.write(f"- **Sources:** {'; '.join(foots)}\n")
                f.write("\n")

            f.write("## Comparative Notes\n\n")
            comp = summary["comparative_analysis"]
            f.write(f"- **License class counts:** {comp['license_classes']}\n")
            f.write(f"- **Usage policy link present:** {comp['has_usage_policy']}\n")
            f.write(f"- **Model card link present:** {comp['has_model_card_link']}\n\n")

            f.write("## Key Findings\n\n")
            for k in summary["key_findings"]:
                f.write(f"- {k}\n")
            f.write("\n## Recommendations\n\n")
            for r in summary["recommendations"]:
                f.write(f"- {r}\n")
            f.write("\n> **Limitations:** Documentation review only; no quantitative safety testing was performed.\n")

        print(f"Saved: {md}")

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Safety & Governance Analysis")
    ap.add_argument("--metadata", type=str, default=None,
                    help="Path to JSON with verified model metadata (optional).")
    ap.add_argument("--output_dir", type=str, default="results/safety",
                    help="Directory to save outputs.")
    args = ap.parse_args()

    models = load_metadata(args.metadata)
    analyzer = SafetyGovernanceAnalyzer(models)
    summary = analyzer.create_safety_summary()
    evaluation = analyzer.run_safety_evaluation()
    analyzer.write_reports(summary, evaluation, args.output_dir)

    # Console summary (short)
    print("\n=== SAFETY & GOVERNANCE SUMMARY (Qualitative) ===")
    for mid, m in analyzer.models.items():
        print(f"- {mid}: license='{m['license']}' class='{m['license_class']}', usage_policy='{m['usage_policy']}'")
    print(f"Outputs written to: {args.output_dir}")

if __name__ == "__main__":
    main()
