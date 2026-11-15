"""Execute only the bounded data recipe with a fixed CPU model/training implementation."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

# -I ignores caller PYTHONPATH. Load just our fixed, stdlib-only contract file.
spec = importlib.util.spec_from_file_location("recipe_contract", Path(__file__).with_name("recipe_spec.py"))
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def label_problem(inputs, labels, vocab_size):
    if inputs.shape != labels.shape:
        return {"code": "label_shape_mismatch", "input_shape": list(inputs.shape), "label_shape": list(labels.shape)}
    # The causal LM loss ignores the first label in each row and shifts the rest.
    targets = labels[:, 1:]
    supervised = targets[targets != -100]
    if not supervised.numel():
        return {"code": "no_supervised_targets", "supervised_targets": 0}
    invalid = supervised[(supervised < 0) | (supervised >= vocab_size)]
    if invalid.numel():
        return {"code": "label_out_of_range", "target": invalid[0].item(), "vocab_size": vocab_size}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    recipe = contract.read_recipe(args.recipe)
    torch.set_num_threads(1)
    torch.manual_seed(431)
    report = {"status": "failed", "stage": "initialize", "device": "cpu", "seed": 431,
        "recipe_sha256": contract.recipe_hash(recipe), "profile": recipe["profile"],
        "pretrained_weights_used": False, "submitted_code_executed": False,
        "model_download_allowed": False, "model_saved": False,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "peft": peft.__version__, "transformers": transformers.__version__}}
    try:
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=recipe["model"]["vocab_size"], hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, tie_word_embeddings=True))
        report["available_linear_modules"] = sorted({name.rsplit(".", 1)[-1] for name, layer in base.named_modules() if isinstance(layer, torch.nn.Linear)})
        report["stage"] = "adapter_injection"
        model = get_peft_model(base, LoraConfig(r=recipe["lora"]["r"], lora_alpha=recipe["lora"]["alpha"],
            target_modules=recipe["lora"]["target_modules"], task_type="CAUSAL_LM", lora_dropout=0.0))
        report["stage"] = "batch_assembly"
        sequences = [torch.tensor(values) for values in recipe["data"]["input_ids"]]
        label_sequences = [torch.tensor(values) for values in (recipe["data"]["labels"] or recipe["data"]["input_ids"])]
        pad = recipe["training"]["collate"] == "pad"
        inputs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0) if pad else torch.stack(sequences)
        labels = torch.nn.utils.rnn.pad_sequence(label_sequences, batch_first=True, padding_value=-100) if pad else torch.stack(label_sequences)
        report.update(input_shape=list(inputs.shape), label_shape=list(labels.shape), stage="forward")
        if recipe["data"]["labels"] is not None:
            problem = label_problem(inputs, labels, recipe["model"]["vocab_size"])
            if problem:
                report["label_problem"] = problem
                raise ValueError("Invalid causal-LM labels: " + problem["code"])
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        before = [parameter.detach().clone() for parameter in trainable]
        base_frozen = all(not parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name)
        optimizer = torch.optim.SGD(trainable, lr=recipe["training"]["learning_rate"])
        model.train()
        losses, finite_gradients = [], True
        for _ in range(recipe["training"]["steps"]):
            optimizer.zero_grad()
            report["stage"] = "forward"
            with contextlib.nullcontext() if recipe["training"]["gradients_enabled"] else torch.no_grad():
                loss = model(input_ids=inputs, attention_mask=inputs.ne(0), labels=labels, use_cache=False).loss
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite loss; this is not a successful validation")
            report.update(stage="backward", loss_requires_grad=loss.requires_grad)
            loss.backward()
            finite_gradients = finite_gradients and bool(trainable) and all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in trainable)
            optimizer.step()
            losses.append(loss.item())
        changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable))
        finite_parameters = all(torch.isfinite(parameter).all().item() for parameter in trainable)
        report.update(status="passed" if changed and finite_gradients and base_frozen and finite_parameters else "failed", stage="completed",
            losses=losses, finite_gradients=finite_gradients, parameters_changed=changed,
            base_parameters_frozen=base_frozen, validation_steps=len(losses),
            finite_parameters=finite_parameters,
            trainable_parameters=sum(parameter.numel() for parameter in trainable))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), log=traceback.format_exc())
    print(json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
