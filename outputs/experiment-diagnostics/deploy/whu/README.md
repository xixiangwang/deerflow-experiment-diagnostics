# 武大超算部署

以 Rocky Linux 9、Python 3.12 与离线 wheel 为例。安装在自己的项目目录，无需 sudo。

外层包 whu-deerflow 包含三个校验过的压缩文件、Python 3.9 可运行的安装器和两个 Slurm 模板。独立 Python 来自 Astral python-build-standalone 的官方发行版本，分发文件按官方 GitHub asset digest 校验；即便从镜像下载，也必须逐字节匹配该哈希。

先只上传外层包。确认付费账号关联后，在解压出的 whu-deerflow 目录通过 sbatch -A 已确认账号 whu-setup.sbatch 提交 CPU 安装作业。它请求 hpib 分区 2 核、8GB 内存、最长 20 分钟，不请求 GPU，也不调用模型 API。解压、离线安装和测试均在分配的计算节点运行。

安装成功后，state/installation.json 给出实际应用路径和框架解释器。该阶段尚未安装 torch，也没有完成 GPU 模型验证。GPU 探测模板单独请求 a100x4 分区 1 卡，必须在确认账号和下一阶段操作后显式提交；不能因为物理节点列出多张卡就认为账户可使用全部卡。

模板使用 UTF-8 无 BOM、LF 换行。按实际 Slurm 账号、分区和配额配置，先完成 CPU 环境检查，再提交单卡任务。

`whu-training-setup.sbatch` 需要自备 Python、离线 wheel 和本地权重，文件格式见同目录安装脚本。已完成的 A100 评测见[评测记录](../../../../docs/evaluation.md)。

官方依据：

- https://docs.hpc.whu.edu.cn/files/whuhpcdocs.wiki/slurm_basic.html
- https://docs.hpc.whu.edu.cn/files/whuhpcdocs.wiki/partition.html
- https://docs.hpc.whu.edu.cn/files/whuhpcdocs.wiki/storage_sftp.html
- https://github.com/astral-sh/python-build-standalone/blob/main/docs/running.rst
