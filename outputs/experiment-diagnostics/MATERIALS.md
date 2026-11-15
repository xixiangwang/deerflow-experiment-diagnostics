# 用户实验材料输入

现在可以提交自己选定的材料，由官方 DeerFlow v2.0.0 + DeepSeek Flash 进行只读诊断。
不再要求预设案例编号。它不会执行或修改提交的代码，也不会声称已经验证修复。

## 输入方式

方式一：UTF-8 JSON 文件，字段如下（字段可省略，但缺失会影响诊断）：

```json
{
  "log": "完整报错及其前后日志",
  "experiment_config": {"batch_size": 2},
  "environment": {"versions": {"torch": "2.5.1+cu121", "peft": "0.21.2"}},
  "code": "相关代码片段，仅作为文本读取",
  "question": "需要助手分析的问题"
}
```

方式二：材料目录，只读取其中名称固定的 log.txt、config.json、environment.json、code.py。
不遍历其他文件、不读取子目录。config.json 对应 experiment_config，environment.json 对应 environment。

在仓库根目录使用框架运行环境：

```bash
python outputs/experiment-diagnostics/launch.py materials --input outputs/experiment-diagnostics/examples/submitted-batch
```

每次新建 reports/materials/ 下的独立目录，保存脱敏输入、模型答案、完整工具记录和可信执行状态。
也可使用[本地页面](CONSOLE.md)。

缺少配置、环境和代码，且只提供一行日志或更少时，程序先做本地检查。若仍无法识别明确现象，直接返回 `insufficient_evidence` 和需要补充的材料。此类报告的 `report_source` 是 `local_evidence_check`，通过 `launch.py materials` 运行时无需输入 API Key。

若模型返回可解析的对象，但 `verification_plan` 是非空字符串，交付报告将整段原文作为数组的一项。`normalized_fields` 记录处理，`raw_answer` 保留模型原文，`strict_contract_passed` 保留严格格式失败。引用和执行状态仍按原有契约校验。

当前报告协议版本 3 包含 references：每条声明的 citation 必须逐字匹配本次工具返回的完整 URL 与行范围，used_for 说明用途。不使用文档时返回空数组。citation_audit 核对来源、保留哈希与版本适用标记；它不检查语义蕴含或引用完整性。

后续运行目录中的 provenance/ 保存当时的场景源码、技能、知识清单与索引哈希，结束时检查运行中是否变化。不补造旧记录的快照，也不包含进程环境变量或模型权重。

## 已实际验证

1. 使用实际 CPU 失败日志、配置、环境与代码摘录组成材料目录，未提供预设编号或修复答案。
   模型识别了变长样本直接 stack 的问题，给出 padding、mask 和验证计划。
2. 仅提交“训练中途停止”、没有日志和配置的 JSON。模型返回证据不足并要求补充信息。

两次都是实际 DeepSeek 调用和官方 DeerFlow 客户端运行，validation_status 均为 not_executed。
这些是输入流程冒烟检查，不属于独立诊断效果评测。原始日志来自开发 fixture，不冒充外部用户事故。

## 边界

- 该路径只开放两个只读工具；通过标准中间件同时过滤工具列表并拒绝其他工具调用。
- 提交代码只做 AST 静态解析，不 import、不 exec；所有验证步骤都是待执行计划。
- 显式提供版本时先精确检索。若没有匹配资料，模型可明确选择其他版本作通用参考，必须标记兼容性未核实。
- 常见密钥格式与配置敏感字段会脱敏，但这不是完备的秘密识别；提交前仍应选择必要材料。
- 支持普通 UTF-8 文件，单文件和合并输入有大小限制，不支持压缩包、任意项目递归扫描或 YAML 对象加载。
- 模型可能过度排除其他原因或给出缺乏匹配版本依据的建议，输出仍需核对，不能视为已完成修复。

受控预设案例的真实修复验证仍通过`launch.py evaluate`运行；不会用预设案例的成功冒充用户实验已验证。
