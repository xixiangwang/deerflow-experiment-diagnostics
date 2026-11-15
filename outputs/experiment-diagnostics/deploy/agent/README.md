# Agent 服务迁移准备

本目录对应真正调用 DeerFlow 的只读材料诊断服务。上一级 deploy/Dockerfile 和旧 diagnostic-server-bundle.tar.gz 对应离线规则 API，两者不同。

## 不使用 Docker 的启动方式

在仓库根目录先运行 `python tools/setup_upstream.py` 准备固定版本的 DeerFlow 源码，再运行以下命令：

```bash
# 必须使用已有 Python 3.12；默认只打印计划，不安装
python3.12 outputs/experiment-diagnostics/bootstrap_runtime.py

# 创建本项目专用虚拟环境并按平台哈希锁安装，随后自动检查
python3.12 outputs/experiment-diagnostics/bootstrap_runtime.py --install

work/runtime-py312/bin/python outputs/experiment-diagnostics/launch.py index

# 提示输入模型密钥；只在调用进程使用
work/runtime-py312/bin/python outputs/experiment-diagnostics/launch.py console --host 127.0.0.1 --port 8890
```

目前锁文件面向 x86_64 Windows 和 glibc Linux、Python 3.12；ARM、musl Linux 与其他 Python 小版本须重新解析验证。依赖版本来自已有验收环境，哈希由包索引生成；这是分发文件校验，不是代码安全审计。

离线安装可在联网环境事先下载并校验锁文件中的 wheel，再通过 `--wheelhouse` 指定本地目录：

```bash
python3.12 outputs/experiment-diagnostics/bootstrap_runtime.py --install --wheelhouse /离线包解压目录/linux-wheelhouse
```

此模式设置 --no-index，只从所选目录查找锁定依赖。下载验收针对 CPython 3.12、x86_64 和 glibc 2.17 至 2.28 的兼容 wheel；实际主机是否兼容仍由其平台和 pip 标签决定，不能在 Windows 上加载这些 Linux 二进制包。

bootstrap 默认使用 pip，也可通过 --uv 指定已有 uv。它拒绝修改没有本项目标记的非空环境、当前 Python 环境和源代码目录，不会修改已有训练环境。已有 torch/peft 环境可通过 --torch-python 指定，额外执行固定 CPU 集成检查；页面本身不需要 torch 或 GPU。

在自己的电脑另开终端转发端口（替换用户名和主机）：

```bash
ssh -L 8890:127.0.0.1:8890 用户名@服务器
```

浏览器打开服务给出的本地地址，访问口令与模型密钥不同。默认前台运行；终端关闭可能终止服务，后台守护由服务器环境决定，当前未配置账户服务或集群调度。

## Docker 备选配置

在已安装 Docker Compose 的 Linux 主机，先在当前 shell 设置 DIAG_API_KEY 与单独的 DIAG_ACCESS_TOKEN；不要写入 Dockerfile 或镜像。在仓库根目录完成源码依赖安装和索引构建后启动：

```bash
docker compose -p deerflow-experiment -f outputs/experiment-diagnostics/deploy/agent/compose.yaml up --build -d
```

Compose 仅映射本机 127.0.0.1:8890，仍可通过 SSH 转发访问。镜像依赖安装使用哈希锁，运行用户为 10001；没有 torch、模型权重或执行提交代码的工具。容器环境变量由 Docker 管理，不要将带密钥的容器配置输出或截图分享。

运行环境放在镜像的 /opt/deerflow-runtime，工作状态和报告分别在两个命名卷，避免升级镜像时被旧持久卷中的虚拟环境遮蔽。Dockerfile 专用忽略文件限制构建上下文，不发送整个 work/。基础镜像使用 python:3.12-slim 标签，未固定镜像 digest；尚未实际构建，不声称完全可复现镜像。

## 重启与数据

任务元数据保存在 work/agent-jobs/{id}/job.json，先写临时文件再原子替换。恢复最近最多 64 份记录；已完成报告可继续经原 id 查询，无需再次调用模型。还在排队或运行的任务会变为 interrupted，执行结果未知，不自动重试。旧工作进程可能已产生报告，人工查看后再决定是否重新提交。

同一任务目录只允许一个服务进程写入；第二个进程启动会明确失败，避免误把另一个活跃进程的任务标为中断。损坏记录会跳过；磁盘写入失败会返回错误或显式提示恢复不可靠。磁盘报告与材料不会自动删除，当前没有磁盘配额或多用户管理。

材料仍只读，验证状态始终为未执行。自动修复、CPU/GPU 验证只在其他固定实验入口提供，不因迁移服务而开放任意代码执行。
