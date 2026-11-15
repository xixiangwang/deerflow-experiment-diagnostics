# 服务器运行说明

当前可部署的是**离线诊断与检索 API**，不是完整 DeerFlow 网页版，也不包含大模型推理服务。

## 运行包

`diagnostic-server-bundle.tar.gz` 包含场景代码、官方扩展 API 源码、22 篇知识资料及索引，不包含本机 Python 环境、模型权重、密钥或完整 DeerFlow 后端。

服务器需已有 Python 3.12+（包括 sqlite3/FTS5），解压后从包根目录运行：

```bash
bash experiment-diagnostics/deploy/run_linux.sh
```

默认监听 `127.0.0.1:8877`。同一台服务器验证：

```bash
curl http://127.0.0.1:8877/health
curl -H 'Content-Type: application/json' \
  -d '{"query":"prepare_model_for_kbit_training","package":"peft","versions":{"peft":"0.13.2"}}' \
  http://127.0.0.1:8877/search
curl -H 'Content-Type: application/json' \
  -d '{"log":"CUDA out of memory","environment":{"versions":{"torch":"2.5.1"}}}' \
  http://127.0.0.1:8877/diagnose
```

通过 SSH 端口转发访问即可；如果明确需要绑定非本机地址，先在服务器设置 `DIAG_ACCESS_TOKEN`，再加 `--host 0.0.0.0`，POST 请求需携带 `Authorization: Bearer ...`。

## Docker 可选入口

在解压后的包根目录执行：

```bash
docker build -f experiment-diagnostics/deploy/Dockerfile -t experiment-diagnostics:offline .
docker run --rm -p 127.0.0.1:8877:8877 -e DIAG_ACCESS_TOKEN experiment-diagnostics:offline
```

此 Dockerfile 尚未实际构建验证；基础镜像需要联网拉取，启动前环境变量 `DIAG_ACCESS_TOKEN` 需有值。

## 后续完整 Agent 部署

完整 DeerFlow 需另外在服务器检出官方固定提交并按其说明安装。场景插件入口是 `experiment_diagnostics:install`，包含两个模型工具：`diagnose_experiment` 和 `search_experiment_knowledge`。

打包安装扩展后，在可信的启动配置 `plugins` 中设置：

```yaml
plugins:
  - use: experiment_diagnostics:install
    required: true
    config:
      knowledge_index: /你的绝对路径/experiment-diagnostics/knowledge/index.sqlite3
```

真实模型还需要支持工具调用的模型端点。上述接口检查不证明模型调用成功。
