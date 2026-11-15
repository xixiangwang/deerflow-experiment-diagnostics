"""Developer retrieval regression set; not independent or model-answer evaluation."""
import json
import hashlib
import time
import uuid
import argparse
from pathlib import Path
from experiment_diagnostics.knowledge import DEFAULT_INDEX, retrieve

ROOT = Path(__file__).resolve().parent
QUERIES = [
    ("kbit", "4bit 量化训练前 prepare_model_for_kbit_training", "peft", "0.13.2", "quantization.md", "prepare_model_for_kbit_training"),
    ("cuda", "显存 empty_cache memory_allocated memory_reserved", "pytorch", "2.5.1", "cuda.rst", "empty_cache"),
    ("amp", "混合精度 GradScaler autocast", "pytorch", "2.5.1", "amp_examples.rst", "GradScaler"),
    ("padding", "padding truncation pad_token", "transformers", "4.46.3", "pad_truncation.md", "padding"),
    ("chat", "chat template apply_chat_template add_generation_prompt", "transformers", "4.46.3", "chat_templating.md", "apply_chat_template"),
    ("lora", "LoRA target_modules all-linear", "peft", "0.13.2", "lora.md", "target_modules"),
    ("collator", "DataCollatorForLanguageModeling mlm False", "transformers", "4.46.3", "data_collator.md", "DataCollatorForLanguageModeling"),
    ("seed", "torch manual_seed reproducibility", "pytorch", "2.5.1", "randomness.rst", "manual_seed"),
    ("numerics", "NaN inf numerical accuracy overflow", "pytorch", "2.5.1", "numerical_accuracy.rst", "inf"),
    ("checkpoint", "save_pretrained adapter_model.safetensors", "peft", "0.13.2", "checkpoint.md", "adapter_model"),
    ("current-kbit", "prepare_model_for_kbit_training", "peft", "0.21.2", "quantization.md", "prepare_model_for_kbit_training"),
    ("current-lora", "target_modules all-linear", "peft", "0.21.2", "lora.md", "target_modules"),
    ("current-custom", "named_modules target_modules custom models", "peft", "0.21.2", "custom_models.md", "named_modules"),
    ("current-qwen2", "Qwen2Attention q_proj k_proj v_proj", "transformers", "5.18.0", "modeling_qwen2.py", "q_proj"),
    ("current-amp", "mixed precision bf16 fp16 autocast", "transformers", "5.18.0", "mixed_precision_training.md", "bf16"),
    ("current-memory", "optimizer states activations gradients memory", "transformers", "5.18.0", "model_memory_anatomy.md", "optimizer"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diversify-sources", action="store_true")
    args = parser.parse_args()
    cases = []
    for identifier, query, package, version, expected_doc, text in QUERIES:
        result = retrieve(query, top_k=3, versions={package: version}, package=package,
                          diversify_sources=args.diversify_sources)
        doc_hit = any(h["url"].endswith("/" + expected_doc) for h in result["hits"])
        evidence_hit = any(h["url"].endswith("/" + expected_doc) and text.lower() in h["text"].lower()
                           for h in result["hits"])
        cases.append({"id": identifier, "query": query, "expected_document": expected_doc,
                      "expected_text": text, "document_hit_at_3": doc_hit,
                      "package": package, "version": version,
                      "all_hits_match_version": all(h["version"] == version and h["applicability"] == "exact_version" for h in result["hits"]),
                      "text_hit_at_3": evidence_hit, "retrieval": result})
    checks = {
        "chitchat_skipped": retrieve("你好")["status"] == "skipped_chitchat",
        "out_of_scope_has_no_hits": not retrieve("学校超算登录口令")["hits"],
        "mismatched_version_has_no_hits": not retrieve("padding truncation", package="transformers",
                                                        versions={"transformers": "99.99.99"})["hits"],
    }
    summary = {"positive_queries": len(cases),
               "document_hits_at_3": sum(c["document_hit_at_3"] for c in cases),
               "all_results_match_requested_versions": all(c["all_hits_match_version"] for c in cases),
               "text_hits_at_3": sum(c["text_hit_at_3"] for c in cases), "boundary_checks": checks}
    report = {"summary": summary, "cases": cases, "model_called": False,
              "scope": "开发检索回归集，16 条已知有答案查询和 3 个边界检查；不是独立集，不衡量最终回答正确率。",
              "index_sha256": hashlib.sha256(DEFAULT_INDEX.read_bytes()).hexdigest(),
              "source_diversity": args.diversify_sources,
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "index": str(DEFAULT_INDEX)}
    folder = ROOT / "reports/retrieval"
    folder.mkdir(exist_ok=True)
    path = folder / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6] + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reports/latest-retrieval.json").write_text(json.dumps({"file": str(path), "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
