"""Fixed multi-fault CPU experiments; no user source or external weights are executed."""
import argparse
import contextlib
import json
import traceback
import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

TRIALS = {"trial-201": {"targets", "grad"}, "trial-202": {"padding", "labels"},
          "trial-203": {"targets", "padding", "grad"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", choices=sorted(TRIALS), required=True)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--target-modules", default='["q_proj","v_proj"]')
    args = parser.parse_args()
    patches, targets = json.loads(args.patches), json.loads(args.target_modules)
    if not isinstance(patches, list) or not set(patches) <= {"set_target_modules", "pad_sequences", "enable_grad", "align_causal_labels"}:
        raise ValueError("Invalid patch set")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 8 or not set(targets) <= {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
        raise ValueError("Invalid target modules")
    defects = TRIALS[args.trial]
    torch.manual_seed(456)
    torch.set_num_threads(1)
    report = {"trial": args.trial, "device": "cpu", "seed": 456,
              "origin": "authored_multifault_fixture", "pretrained_weights_used": False,
              "applied_patches": patches, "status": "failed", "stage": "initialize",
              "versions": {"torch": torch.__version__, "peft": peft.__version__, "transformers": transformers.__version__}}
    try:
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=40, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32))
        report["available_linear_modules"] = sorted({n.rsplit(".", 1)[-1] for n, p in base.named_modules()
                                                     if isinstance(p, torch.nn.Linear)})
        current_targets = ["unknown_projection"] if "targets" in defects and "set_target_modules" not in patches else targets
        report.update(stage="adapter_injection", target_modules=current_targets)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=current_targets,
                                               task_type="CAUSAL_LM", lora_dropout=0.0))
        samples = [[2, 3, 4, 5, 6], [7, 8, 9]] if "padding" in defects else [[2, 3, 4, 5, 6]]
        report.update(stage="batch_assembly", input_samples=samples, sample_lengths=list(map(len, samples)))
        sequences = [torch.tensor(sample) for sample in samples]
        inputs = (torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
                  if "pad_sequences" in patches else torch.stack(sequences))
        labels = inputs.clone()
        labels[inputs == 0] = -100
        if "labels" in defects and "align_causal_labels" not in patches:
            labels[0, -1] = 77
        report.update(stage="forward", input_shape=list(inputs.shape), labels=labels.tolist(), vocab_size=40)
        trainable = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in trainable]
        optimizer = torch.optim.SGD(trainable, lr=0.05)
        losses = []
        finite_gradients = True
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            disabled = "grad" in defects and "enable_grad" not in patches
            with torch.no_grad() if disabled else contextlib.nullcontext():
                loss = model(input_ids=inputs, attention_mask=inputs.ne(0), labels=labels, use_cache=False).loss
            report.update(stage="backward", loss_requires_grad=loss.requires_grad)
            assert torch.isfinite(loss)
            loss.backward()
            finite_gradients = finite_gradients and all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable)
            optimizer.step()
            losses.append(loss.item())
        changed = any(not torch.equal(a, b.detach()) for a, b in zip(before, trainable))
        report.update(status="passed" if changed and finite_gradients else "failed", stage="completed",
                      finite_gradients=finite_gradients, parameters_changed=changed, losses=losses,
                      validation_steps=2, trainable_parameters=sum(p.numel() for p in trainable))
    except (ValueError, RuntimeError, IndexError, AssertionError) as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), log=traceback.format_exc())
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
