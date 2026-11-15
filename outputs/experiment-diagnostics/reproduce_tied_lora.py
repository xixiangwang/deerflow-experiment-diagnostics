"""CPU check inspired by PEFT #3929, using local random weights and a fixed script."""
import argparse
import json
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


def check(dtype, cast_adapters):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True)
    base = LlamaForCausalLM(config).to(dtype)
    lora = LoraConfig(target_modules=["embed_tokens", "lm_head"], ensure_weight_tying=True, init_lora_weights=False)
    model = get_peft_model(base, lora, autocast_adapter_dtype=cast_adapters)
    embedding = model.base_model.model.model.embed_tokens.lora_embedding_B["default"]
    head = model.base_model.model.lm_head.lora_A["default"].weight
    trainable = [p for p in model.parameters() if p.requires_grad]
    previous = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.SGD(trainable, lr=1.0)
    ids = torch.randint(0, 64, (2, 8))
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    gradients_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable)
    optimizer.step()
    difference = (head.float() - embedding.float().T).abs().max().item()
    return {"base_dtype": str(dtype), "autocast_adapter_dtype": cast_adapters, "adapter_dtype": str(head.dtype),
            "shared_storage": head.data_ptr() == embedding.data_ptr(), "max_difference_after_step": difference,
            "finite_loss": torch.isfinite(loss).item(), "finite_gradients": gradients_finite,
            "trainable_parameters_updated": any(not torch.equal(old, new.detach()) for old, new in zip(previous, trainable))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [check(torch.float32, True), check(torch.bfloat16, True), check(torch.bfloat16, False)]
    pattern_matches = (rows[0]["shared_storage"] and rows[0]["max_difference_after_step"] == 0
                       and not rows[1]["shared_storage"] and rows[1]["max_difference_after_step"] > 0
                       and rows[2]["shared_storage"] and rows[2]["max_difference_after_step"] == 0)
    report = {"source": "https://github.com/huggingface/peft/issues/3929", "device": "cpu", "optimizer_steps_per_case": 1,
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
              "source_environment_recreated": False, "downloaded_model": False, "submitted_code_executed": False,
              "all_runs_finite_and_updated": all(row["finite_loss"] and row["finite_gradients"] and row["trainable_parameters_updated"] for row in rows),
              "observed_pattern_matches_public_report": pattern_matches, "cases": rows}
    text = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
