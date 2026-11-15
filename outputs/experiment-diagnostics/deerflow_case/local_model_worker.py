"""Local Transformers provider worker. Preserves the full supplied prompt, no downloads."""
import json
import re
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

request = json.load(sys.stdin)
started = time.perf_counter()
device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1600000000 else "cpu"
tokenizer = AutoTokenizer.from_pretrained(request["model_path"], local_files_only=True, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(request["model_path"], local_files_only=True,
    trust_remote_code=False, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
prompt = tokenizer.apply_chat_template(request["messages"], tools=request["tools"],
                                       tokenize=False, add_generation_prompt=True)
if not isinstance(prompt, str):
    raise TypeError(f"Unexpected chat-template result: {type(prompt).__name__}")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
if inputs["input_ids"].shape[1] > 12000:
    raise ValueError("Local smoke test input exceeds 12000 tokens; no silent truncation")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
calls = []
for number, block in enumerate(re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.S)):
    try:
        value = json.loads(block)
        arguments = value.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if isinstance(value.get("name"), str) and isinstance(arguments, dict):
            calls.append({"name": value["name"], "args": arguments, "id": f"local-call-{number}"})
    except (ValueError, TypeError):
        pass  # Preserve malformed output as text; never synthesize a tool action.
content = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.S).strip() if calls else raw
print(json.dumps({"content": content, "tool_calls": calls, "raw": raw,
                  "input_tokens": inputs["input_ids"].shape[1], "device": device,
                  "seconds": time.perf_counter() - started}, ensure_ascii=False), flush=True)
