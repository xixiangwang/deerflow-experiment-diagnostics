"""Fixed CPU developer cases. No submitted source, model downloads, or shell execution."""
import argparse
import contextlib
import json
import traceback
import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

KINDS = {"exp-101": "targets", "exp-102": "padding", "exp-103": "grad",
         "exp-104": "labels", "exp-105": "normal"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", choices=sorted(KINDS), required=True)
    parser.add_argument("--mode", choices=["inspect", "validate"], required=True)
    parser.add_argument("--fix-type", default="none")
    parser.add_argument("--target-modules", default='["q_proj","v_proj"]')
    args = parser.parse_args()
    kind = KINDS[args.case_id]
    torch.manual_seed(123)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    model = Qwen2ForCausalLM(config)
    modules = sorted({name.rsplit(".", 1)[-1] for name, layer in model.named_modules()
                      if isinstance(layer, torch.nn.Linear)})
    targets = ["not_a_real_module"] if kind == "targets" else ["q_proj", "v_proj"]
    if args.fix_type == "set_target_modules":
        targets = json.loads(args.target_modules)
        allowed = set(modules) - {"lm_head"}
        if not isinstance(targets, list) or not 1 <= len(targets) <= 8 or any(t not in allowed for t in targets):
            raise ValueError("Unsupported target modules")
    samples = [[1, 2, 3, 4], [5, 6]] if kind == "padding" else [[1, 2, 3, 4]]
    disable_grad = kind == "grad" and args.fix_type != "enable_grad"
    report = {"case_id": args.case_id, "device": "cpu", "seed": 123,
              "task_type": "CAUSAL_LM", "vocab_size": 32,
              "versions": {"torch": torch.__version__, "peft": peft.__version__,
                           "transformers": transformers.__version__},
              "configuration": {"target_modules": targets, "grad_enabled": not disable_grad},
              "input_samples": samples, "sample_lengths": list(map(len, samples)),
              "available_linear_modules": modules, "fix_type": args.fix_type,
              "pretrained_weights_used": False, "origin": "authored_cpu_developer_fixture"}
    try:
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets,
                                                task_type="CAUSAL_LM", lora_dropout=0.0))
        sequences = [torch.tensor(values) for values in samples]
        if args.fix_type == "pad_sequences":
            inputs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        else:
            inputs = torch.stack(sequences)
        labels = inputs.clone()
        labels[inputs == 0] = -100
        if kind == "labels" and args.fix_type != "restore_label_alignment":
            labels[0, -1] = 99
        report["labels"] = labels.tolist()
        report["input_shape"] = list(inputs.shape)
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        optimizer = torch.optim.SGD(trainable, lr=0.05)
        losses = []
        finite_gradients = True
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            context = torch.no_grad() if disable_grad else contextlib.nullcontext()
            with context:
                loss = model(input_ids=inputs, attention_mask=inputs.ne(0), labels=labels, use_cache=False).loss
            assert torch.isfinite(loss)
            loss.backward()
            finite_gradients = finite_gradients and all(
                p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable)
            optimizer.step()
            losses.append(loss.item())
        changed = any(not torch.equal(old, p.detach()) for old, p in zip(before, trainable))
        report.update(status="passed" if finite_gradients and changed else "failed", losses=losses,
                      finite_gradients=finite_gradients, parameters_changed=changed, validation_steps=2,
                      trainable_parameters=sum(p.numel() for p in trainable))
    except (RuntimeError, ValueError, IndexError, AssertionError) as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), log=traceback.format_exc())
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
