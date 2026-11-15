"""Fixed local pretrained-model GPU trial, two steps maximum, no model saving/downloads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
import torch
import transformers
import peft
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

GIB = 1024 ** 3
OOM_PROFILES = {"local": (256, 2.40), "a100": (2048, 10)}
EFFECTIVE_BATCH = 8

profile_spec = importlib.util.spec_from_file_location("model_profile", Path(__file__).with_name("model_profile.py"))
profile_module = importlib.util.module_from_spec(profile_spec)
profile_spec.loader.exec_module(profile_module)


def configure_oom_trial(profile, variant, report, repair_micro_batch):
    sequence_length, cap_gib = OOM_PROFILES[profile]
    micro_batch = EFFECTIVE_BATCH if variant == "baseline" else repair_micro_batch
    report.update(oom_profile=profile, oom_variant=variant, allocator_cap_bytes=int(cap_gib * GIB),
                  micro_batch_size=micro_batch, effective_batch_size=EFFECTIVE_BATCH,
                  gradient_accumulation_steps=EFFECTIVE_BATCH // micro_batch,
                  sequence_length=sequence_length, input_seed=789)
    free, total = torch.cuda.mem_get_info()
    if profile == "a100" and (torch.cuda.device_count() != 1 or "A100" not in report["gpu"]):
        raise RuntimeError("The A100 profile requires exactly one visible A100")
    if free < report["allocator_cap_bytes"] + 256 * 1024 ** 2:
        raise RuntimeError("Insufficient free CUDA memory for the controlled allocator-budget trial")
    if torch.cuda.get_allocator_backend() != "native":
        raise RuntimeError("The controlled allocator-budget trial requires the native CUDA allocator")
    torch.cuda.set_per_process_memory_fraction(report["allocator_cap_bytes"] / total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect", "validate", "preflight"], required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--target-modules", default='["q_proj","v_proj"]')
    parser.add_argument("--oom-profile", choices=OOM_PROFILES)
    parser.add_argument("--oom-variant", choices=["baseline", "repair"])
    parser.add_argument("--repair-micro-batch", type=int, choices=[1, 2, 4])
    args = parser.parse_args()
    if bool(args.oom_profile) != bool(args.oom_variant) or (args.oom_profile and args.mode != "validate"):
        parser.error("The OOM profile and variant must be supplied together in validate mode")
    if args.repair_micro_batch is not None and args.oom_variant != "repair":
        parser.error("--repair-micro-batch requires the OOM repair variant")
    report = {"mode": args.mode, "status": "failed", "device": "cuda", "pretrained_weights_used": False,
              "execution_started": False, "stage": "profile_validation",
              "save_pretrained_called": False, "model_download_allowed": False,
              "versions": {"torch": torch.__version__, "peft": peft.__version__, "transformers": transformers.__version__}}
    started = time.perf_counter()
    weight = args.model / "model.safetensors"
    before = (weight.stat().st_size, weight.stat().st_mtime_ns) if weight.is_file() else None
    try:
        report["model_profile"] = profile_module.inspect_local_profile(args.model)
        if args.mode == "preflight":
            report.update(status="ready", device="not_loaded", cuda_available=torch.cuda.is_available(), validation_steps=0)
            return
        if not torch.cuda.is_available():
            report.update(status="skipped", reason="No CUDA device")
            return
        free, total = torch.cuda.mem_get_info()
        report.update(gpu=torch.cuda.get_device_name(0), free_bytes_before=free, total_bytes=total)
        if free < 1900000000:
            report.update(status="skipped", reason="Less than 1.9GB free; will not start")
            return
        targets = json.loads(args.target_modules)
        allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        if not isinstance(targets, list) or not 1 <= len(targets) <= 8 or any(t not in allowed for t in targets):
            raise ValueError("Unsupported module names")
        torch.manual_seed(789)
        torch.cuda.manual_seed_all(789)
        if args.oom_profile:
            report["stage"] = "allocator_setup"
            configure_oom_trial(args.oom_profile, args.oom_variant, report, args.repair_micro_batch or 1)
        torch.cuda.reset_peak_memory_stats()
        # PyTorch 2.5.1 includes emulation by default; V100 must use native FP16.
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        report.update(compute_capability=list(torch.cuda.get_device_capability(0)),
                      native_bf16_supported=native_bf16)
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        report["dtype"] = str(dtype)
        report.update(stage="model_loading", execution_started=True)
        if not args.oom_profile:
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                                  torch_dtype=dtype, use_safetensors=True).to("cuda")
        report.update(stage="adapter_injection", pretrained_weights_used=True)
        modules = sorted({name.rsplit(".", 1)[-1] for name, layer in base.named_modules()
                          if isinstance(layer, torch.nn.Linear)})
        report.update(available_linear_modules=modules, model_type=base.config.model_type,
                      config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest())
        selected = ["not_an_existing_projection"] if args.mode == "inspect" else targets
        report["target_modules"] = selected
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=selected,
                                               task_type="CAUSAL_LM", lora_dropout=0.0))
        if args.mode == "inspect":
            raise RuntimeError("Broken pretrained fixture unexpectedly succeeded")
        report["stage"] = "training"
        if args.oom_profile:
            generator = torch.Generator(device="cpu").manual_seed(report["input_seed"])
            tokens = torch.randint(1000, 16000, (EFFECTIVE_BATCH, report["sequence_length"]), generator=generator)
            report["input_sha256"] = hashlib.sha256(bytes(tokens.untyped_storage())).hexdigest()
            batch = {"input_ids": tokens.to("cuda")}
            labels = batch["input_ids"]
            micro_batch = report["micro_batch_size"]
        else:
            tokenizer.pad_token = tokenizer.eos_token
            batch = tokenizer(["实验记录：检查配置并记录结果。", "训练日志应保留错误与验证结果。"],
                              padding=True, truncation=True, max_length=32, return_tensors="pt").to("cuda")
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            micro_batch = len(labels)
        trainable = [p for p in model.parameters() if p.requires_grad]
        previous = [p.detach().clone() for p in trainable]
        optimizer = torch.optim.SGD(trainable, lr=0.001)
        model.train()
        losses, gradients_finite = [], True
        report["validation_steps"] = 0
        accumulation = len(labels) // micro_batch
        for step in range(2):
            optimizer.zero_grad()
            step_loss = 0.0
            for start in range(0, len(labels), micro_batch):
                selection = slice(start, start + micro_batch)
                inputs = {key: value[selection] for key, value in batch.items()}
                report["stage"] = "training_forward" if args.oom_profile else "training"
                loss = model(**inputs, labels=labels[selection], use_cache=False).loss / accumulation
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite training loss")
                report["stage"] = "training_backward" if args.oom_profile else "training"
                loss.backward()
                step_loss += loss.item()
                del loss
            gradients_finite = gradients_finite and all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable)
            report["stage"] = "optimizer_step" if args.oom_profile else "training"
            optimizer.step()
            report["validation_steps"] = step + 1
            losses.append(step_loss)
        changed = any(not torch.equal(old, parameter.detach()) for old, parameter in zip(previous, trainable))
        parameters_finite = all(torch.isfinite(parameter).all().item() for parameter in trainable)
        report.update(status="passed" if changed and gradients_finite and parameters_finite else "failed", validation_steps=2,
                      stage="completed",
                      gradients_finite=gradients_finite, parameters_changed=changed, losses=losses,
                      parameters_finite=parameters_finite,
                      input_shape=list(batch["input_ids"].shape), trainable_parameters=sum(p.numel() for p in trainable),
                      base_parameters_frozen=all(not p.requires_grad for name, p in model.named_parameters() if "lora_" not in name))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), log=traceback.format_exc())
    finally:
        if torch.cuda.is_available():
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if args.oom_profile:
                report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        after = (weight.stat().st_size, weight.stat().st_mtime_ns) if weight.is_file() else None
        report["original_weight_stat_unchanged"] = before == after and before is not None
        report["seconds"] = time.perf_counter() - started
        if args.oom_profile:
            report["scope"] = "固定进程分配器上限下的 CUDA 显存实验，使用合成 token 和两步优化。"
        else:
            report["scope"] = ("仅配置与文件元数据预检；未加载权重、未验证 GPU 训练。" if args.mode == "preflight" else
                               "固定现有 0.5B 权重的短程 LoRA GPU 验证；不证明任务质量、收敛或泛化。")
        print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
