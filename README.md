# 基于 DeerFlow 的大模型实验诊断与验证助手

给定训练日志、环境或实验配置，助手会检查故障、检索对应版本的官方文档并输出诊断。对预定义的实验配置，Agent 可以选择受限修改并重跑训练，以运行回执验证修复结果。项目复用 DeerFlow 2.0 的 Agent 循环、Skill 和工具接口，诊断工具、状态管理及评测由本仓库实现。

| 场景 | 输入与结果 |
| --- | --- |
| 材料诊断 | 提交日志、JSON 或代码摘录，返回带引用的只读诊断与验证计划 |
| 受控修复 | 针对预定义训练配置检查 LoRA 投影层、张量形状、标签和梯度，复跑时核对修订号、运行 ID 和配置哈希 |
| GPU 验证 | 用已有本地权重或 Slurm 单卡作业复现指定故障，记录显存、梯度、适配器更新与基座冻结 |

## 快速开始

在仓库根目录运行。框架的哈希锁安装支持 Python 3.12、Windows x86-64 和 glibc Linux x86-64。

```bash
python tools/setup_upstream.py
python outputs/experiment-diagnostics/bootstrap_runtime.py --install
```

激活 `work/runtime-py312`，或直接使用其中的 Python（Windows 为 `Scripts/python.exe`，Linux 为 `bin/python`）。

```bash
python outputs/experiment-diagnostics/launch.py index
python outputs/experiment-diagnostics/launch.py recipe-check --input outputs/experiment-diagnostics/examples/recipes/stack-and-targets.json
python outputs/experiment-diagnostics/launch.py tests
```

带模型的命令使用本地进程环境变量 `DIAG_API_KEY`，也可在终端提示时输入。默认使用 DeepSeek，接口和模型可通过 `--base-url` 与 `--model` 配置。

```bash
python outputs/experiment-diagnostics/launch.py materials --input outputs/experiment-diagnostics/examples/submitted-batch
python outputs/experiment-diagnostics/launch.py console
```

控制台默认监听 `127.0.0.1:8890`。材料入口把日志和代码作为文本分析。受控复跑执行项目内固定 worker。

```bash
python outputs/experiment-diagnostics/launch.py recipe --input outputs/experiment-diagnostics/examples/recipes/stack-and-targets.json --torch-python /path/to/training/python
```

训练 Python 需安装 PyTorch、Transformers、PEFT。GPU 示例还需 CUDA 及已有本地模型权重，可通过 `--local-model` 指定。程序生成的报告保存在 `outputs/experiment-diagnostics/reports/`。

## 评测

以同一模型完成 12 个合成留出任务的三路对照，共 36 次运行。直接回答、轻量工具 Agent、DeerFlow 分别通过 **8/12、12/12、10/12**，两组工具 Agent 均修复全部 **7/7** 个可修复案例。标签保护与检索限额更新后，原失败案例回归 **2/2**，新增合成案例验收 **6/6**。

A100 单卡实验在 10 GiB 进程分配器上限下用预设方案复现前向 OOM，保持有效批量并将微批从 8 降到 1，完成两步 BF16 LoRA。峰值分配显存为 **6.68 GiB**。

- [评测协议、真实问题复核与边界](docs/evaluation.md)
- [训练配置协议和修复动作](outputs/experiment-diagnostics/RECIPES.md)
- [材料输入协议](outputs/experiment-diagnostics/MATERIALS.md)
- [第三方来源与许可](THIRD_PARTY.md)
