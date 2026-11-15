"""Use an existing local model in an existing torch environment; no downloads."""
import argparse
import json
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads((ROOT / "reports/local-request.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1600000000 else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               trust_remote_code=False, torch_dtype=dtype).to(device).eval()
    messages = [{"role": "user", "content": request["prompt"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1500).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    generate_start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=160, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    if device == "cuda":
        torch.cuda.synchronize()
    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    report = {"model": str(args.model.resolve()), "device": device,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "generation_seconds": time.perf_counter() - generate_start,
              "total_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "input_tokens": inputs["input_ids"].shape[1],
              "answer": answer, "question": request["question"],
              "smoke_check": {"required_function_mentioned": "prepare_model_for_kbit_training" in answer,
                              "note": "字符串检查仅用于发现明显空答，不能证明回答正确。"},
              "sources": [{k: s[k] for k in ("label", "citation", "version", "source_sha256")}
                          for s in request["sources"]],
              "scope": request["scope"], "deerflow_agent_used": False, "model_called": True,
              "quality_evaluated": False}
    (ROOT / "reports/local-generation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
