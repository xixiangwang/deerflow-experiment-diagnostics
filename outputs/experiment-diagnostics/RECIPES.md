# 选定 JSON 配置的受限实验重现

新增 recipe 入口接受操作者选定的 JSON 配置与数值小批数据。官方 DeerFlow 调用检查、检索和配置修改工具，通过固定 CPU 脚本真实重跑，生成初始与最终配置、错误轨迹和执行证据。

它支持用户自行构造最小重现，不再只接受预设案例编号。当前执行配置固定为 tiny-qwen2-cpu：随机初始化的微型 Qwen2、单层、CPU、LoRA 与 SGD。数据和超参数可以在限制内修改；没有下载预训练权重或保存训练权重。

## 先检查，再诊断

从包根目录运行：

```bash
# 数据协议与资源范围检查，不调用模型 API，也不运行训练
python outputs/experiment-diagnostics/launch.py recipe-check --input outputs/experiment-diagnostics/examples/recipes/stack-and-targets.json

# 官方 DeerFlow + 模型工具调用 + 实际 CPU 验证
python outputs/experiment-diagnostics/launch.py recipe --input outputs/experiment-diagnostics/examples/recipes/stack-and-targets.json --torch-python /path/to/training/python
```

使用已经安装框架依赖的 Python。训练环境通过 `--torch-python` 指定，需要 PyTorch、Transformers 和 PEFT。

需要保留提供的标签时，在 `launch.py recipe` 加 `--preserve-labels`。当 worker 确认实际标签张量与输入形状不匹配、参与损失的标签越界，或移位后没有监督标签时，程序会结束诊断并返回 `unresolved`。报告包含该次失败的运行号、配置版本和哈希，`report_source` 标记为 `worker_label_policy`。此时命令退出码 0 表示报告与执行证据通过核对，训练结论以报告的 `status` 为准。

检索每个配置版本最多两次，每个实验最多四次。同一版本下重复查询消耗一次尝试，并提示模型查看已有检索结果。运行通过后，下一次模型调用用于生成结论。总调用预算为十次，最后一次用于报告。

模型结论缺失或与已通过的回执不符时，程序可以据回执交付结论，`report_source` 标记为 `worker_verified_fallback`。`raw_answer` 和 `format` 记录原始输出，`delivered_strict_json` 记录交付格式。兜底报告在 `strict_output_passed` 中记为未通过。

只检查 JSON 的独立入口无需框架模型依赖：python check_recipe.py 选定文件.json。

## JSON 示例

```json
{
  "schema_version": 1,
  "name": "my-minimal-reproduction",
  "profile": "tiny-qwen2-cpu",
  "model": {"vocab_size": 48},
  "lora": {"r": 3, "alpha": 6, "target_modules": ["missing_projection"]},
  "training": {"steps": 2, "learning_rate": 0.025, "gradients_enabled": true, "collate": "stack"},
  "data": {"input_ids": [[2, 5, 7, 9, 12, 14], [3, 6, 8, 10]]}
}
```

协议只允许这些字段。模型词表 16..512，LoRA r 1..8、alpha 1..32，训练 2..3 步、学习率 0.000001..0.1；最多 4 条样本，每条 3..64 个整数 token。0 保留作 padding，输入 token 必须在词表内。可选 labels 用于重现标签错误；它不由协议检查宣称语义正确。

这批数值 token 是最小化重现的数据格式，不直接接受原模型的大词表 token 或完整数据集。使用自己的数据时需先在支持范围内构造最小重现。

## 可修改什么

- set_lora_targets：选择本次实际观察到的内部模块名，排除 lm_head。
- pad_batch：改为 padding，labels 的 padding 为 -100。
- enable_gradients：启用反向传播所需梯度。
- rebuild_causal_labels：放弃传入 labels，改用输入 token 的因果语言建模标签。

最后一项会改变标签策略，报告的 label_policy_changed 必须明确为 true。它证明更改后的配置能运行，不证明原标签的语义意图错误。对自定义标注任务不可直接把这项替换当作正确修复。

每次只改一项并累积保留，expected_revision 过期时不执行，基础设施失败回滚配置。最多七次实际运行；通过后禁止继续改动。中间错误变化仍是失败，不能宣称验证完成。

## 报告与可信状态

每次在 reports/recipes/ 下生成独立目录，包含 input-recipe.json、final-recipe.json、report.json、answer.md 和 provenance/ 源码快照。原始选定 JSON 不被修改。

成功同时要求：真实退出码 0、有限梯度和参数、adapter 参数更新、基础参数冻结、执行步数在范围内，以及模型报告的 run_id、revision、配置 SHA-256 与最后实际运行一致。健康初始配置只允许 verified_baseline，不应为了展示修复而修改它。

本入口不接收代码、shell、解释器或模型路径字段，不执行用户代码。它属于结构化配置的受限运行；网页材料诊断仍只读。任意用户工程的补丁写入和执行尚未实现，子进程也不是安全沙箱。

三份示例是开发样例。worker 检查中的修改由开发脚本选择，真实 Agent 检查单独记录；不把它们合并为独立诊断准确率。
