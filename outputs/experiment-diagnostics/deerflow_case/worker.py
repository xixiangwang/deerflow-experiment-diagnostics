"""Fixed, tiny CPU LoRA fixture; never runs submitted source or shell commands."""
import argparse
import json
import traceback
import torch
import peft
import transformers
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect", "validate"], required=True)
    parser.add_argument("--target-modules", default='["not_a_real_module"]')
    args = parser.parse_args()
    targets = json.loads(args.target_modules)
    if not isinstance(targets, list) or not 1 <= len(targets) <= 8 or any(
            not isinstance(value, str) for value in targets):
        raise ValueError("Invalid target modules")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.0))
    modules = sorted({name.rsplit(".", 1)[-1] for name, layer in base.named_modules()
                      if isinstance(layer, torch.nn.Linear)})
    report = {"fixture": "tiny_qwen2_lora", "device": "cpu", "seed": 42,
              "versions": {"torch": torch.__version__, "peft": peft.__version__,
                           "transformers": transformers.__version__},
              "available_linear_modules": modules, "requested_target_modules": targets,
              "origin": "authored_fixture_not_user_training_run", "pretrained_weights_used": False}
    if args.mode == "validate" and (not set(targets) <= set(modules) or "lm_head" in targets):
        raise ValueError("Choose observed internal linear module names, excluding lm_head")
    try:
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=targets,
                                               task_type="CAUSAL_LM", lora_dropout=0.0))
    except ValueError as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      log=traceback.format_exc())
        print(json.dumps(report, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    if args.mode == "inspect":
        raise RuntimeError("Broken fixture unexpectedly succeeded")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    losses = []
    gradients_finite = True
    model.train()
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in trainable)
        optimizer.step()
        losses.append(loss.item())
    changed = any(not torch.equal(old, parameter.detach()) for old, parameter in zip(before, trainable))
    report.update(status="passed" if gradients_finite and changed else "failed",
                  trainable_parameters=sum(p.numel() for p in trainable),
                  losses=losses, finite_gradients=gradients_finite, parameters_changed=changed,
                  validation_steps=2, scope="Only fixture correctness, not convergence or task quality")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
