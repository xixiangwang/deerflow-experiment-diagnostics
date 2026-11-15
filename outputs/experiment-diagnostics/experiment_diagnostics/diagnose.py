"""Small explicit checks; uncertain root causes stay uncertain."""
import re


def diagnose(payload):
    log = payload.get("log", "")
    config = payload.get("config", {})
    environment = payload.get("environment", {})
    if not isinstance(log, str) or len(log) > 100000:
        raise ValueError("log must be a string of at most 100000 characters")
    if not isinstance(config, dict) or not isinstance(environment, dict):
        raise ValueError("config and environment must be objects")
    findings = []

    def add(code, cause, evidence, suggestion, verification, certainty="hypothesis"):
        findings.append(dict(code=code, cause=cause, evidence=evidence,
                             suggestion=suggestion, verification=verification,
                             certainty=certainty))

    lines = log.splitlines()
    for number, line in enumerate(lines, 1):
        evidence = [{"source": "log", "line": number, "text": line}]
        match = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", line)
        if match:
            add("missing_module", "当前解释器无法导入模块 " + match.group(1), evidence,
                "核对 sys.executable 与项目依赖清单；确认模块对应的发行包后，在该环境安装锁定依赖。",
                "使用同一个解释器执行导入检查，再重跑原命令。", "observed")
        if "CUDA out of memory" in line:
            add("cuda_oom", "日志报告 CUDA 显存分配失败；具体占用来源仍需检查。", evidence,
                "记录显存占用、batch size、序列长度；先降低单卡 batch size，再检查其他进程。",
                "在相同模型与数据下记录修改前后峰值显存和原失败步骤是否完成。")
        if "ZeroDivisionError" in line and type(config.get("batch_size")) is int and config["batch_size"] == 0:
            add("invalid_batch_size", "batch_size=0 与除零错误同时出现，需要定位分母。",
                evidence + [{"source": "config", "key": "batch_size", "value": 0}],
                "将 batch_size 改成正整数，并在计算训练步数前添加参数校验。",
                "保留其他配置，只修改 batch_size；重跑并确认训练步数为正。")
        if "FileNotFoundError" in line:
            add("missing_file", "输入文件路径无法解析。", evidence,
                "核对日志中的路径、当前工作目录和数据准备步骤；容器内同时检查挂载路径。",
                "在相同工作目录确认文件存在，再重跑原命令。", "observed")
        if re.search(r"\bloss\s*[:=]\s*[-+]?(nan|inf)\b", line, re.I):
            add("nonfinite_loss", "loss 出现非有限值；日志不足以认定是学习率导致。", evidence,
                "先检查输入、标签、梯度与混合精度溢出，再对学习率做单变量对照。",
                "固定随机种子和数据批次，记录首次出现非有限值的张量与步骤。")
        if re.search(r"Target -?\d+ is out of bounds", line):
            add("target_out_of_bounds", "损失计算中的目标类别超出有效范围。", evidence,
                "核对 logits 的类别维度、标签最小值和最大值，以及 ignore_index。",
                "对同一批输入打印类别维度与标签范围；修复标签映射后重算 loss。", "observed")
        if "does not require grad and does not have a grad_fn" in line:
            add("no_grad", "反向传播的张量未关联可求导计算图。", evidence,
                "核对 requires_grad、no_grad 上下文、detach 和 LoRA 可训练参数。",
                "在同一最小批次检查 loss.requires_grad，并确认目标参数获得有限梯度。", "observed")
        if "stack expects each tensor to be equal size" in line:
            add("shape_mismatch", "组装批次时输入张量尺寸不一致。", evidence,
                "检查样本长度与 data collator 的 padding 策略；不要直接堆叠不同长度序列。",
                "记录每条样本形状，填充后确认 batch 形状和 attention_mask 正确。", "observed")
    return {"mode": "offline_rules", "findings": findings,
            "status": "findings" if findings else "insufficient_evidence",
            "limitation": "规则检查只覆盖列出的模式；未发现问题不代表实验正确。修复需实际验证。"}
