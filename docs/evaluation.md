# 评测与复现

## 受控 CPU 修复

冻结集位于 `outputs/experiment-diagnostics/evaluation_materials/recipe-holdout-v1.json`。12 例包括 7 个可修复故障、2 个正常配置、3 个要求保留原始标签的故障。三种方法均使用相同模型和首次真实执行结果。直接回答提交一次修复建议，由评测器回放。轻量工具 Agent 和 DeerFlow 使用相同修复工具、场景 Skill 及最多 10 次模型调用。

| 方法 | 总通过 | 可修复 | 正常配置保持 | 标签约束合规 |
| --- | ---: | ---: | ---: | ---: |
| 直接回答 | 8/12 | 5/7 | 2/2 | 1/3 |
| 轻量工具 Agent | 12/12 | 7/7 | 2/2 | 3/3 |
| DeerFlow | 10/12 | 7/7 | 2/2 | 1/3 |

总分同时检查必要修改、标签约束、执行终态及报告与回执的一致性。严格 JSON 单独计分。轻量工具 Agent 在这一组通过数最高。本组仅覆盖项目内预设故障类型。

先使用实际 CPU worker 预检，再运行模型对照或策略复验。

```bash
python outputs/experiment-diagnostics/launch.py recipe-holdout-preflight --torch-python /path/to/training/python
python outputs/experiment-diagnostics/launch.py recipe-holdout --torch-python /path/to/training/python
python outputs/experiment-diagnostics/launch.py recipe-policy-check --torch-python /path/to/training/python
```

策略复验覆盖检索循环、受保护标签、正常配置及授权重建标签。改进后两个原失败案例通过 2/2，六个新增合成案例通过 6/6。部分案例由代码核对回执后补交结构化报告。

## 真实开源问题

材料位于 `outputs/experiment-diagnostics/evaluation_materials/public-issues-v1.json`，包含 6 个开源问题和 2 个仅报错样例。评测走只读诊断，回看建议是否与版本证据兼容。补充源码后定向复测两例，其中一例仍误判。引用匹配和建议正确分别检查。

```bash
python outputs/experiment-diagnostics/launch.py compare-public
```

## A100 单卡

使用固定输入与有效批量。微批为 8 时，在 10 GiB PyTorch 分配器上限下触发前向 OOM；预设复跑改为微批 1、梯度累积 8。2048 token 输入下完成两步 BF16 LoRA，峰值分配显存 6.68 GiB。验证项包括有限梯度、适配器更新和基座参数冻结。该实验衡量受控配置能否完成两步训练。
