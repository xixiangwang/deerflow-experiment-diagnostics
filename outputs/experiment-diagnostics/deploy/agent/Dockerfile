# Build context is the extracted source package root; this is the read-only Agent console.
FROM python:3.12-slim
WORKDIR /app
COPY outputs/experiment-diagnostics/deploy/agent/requirements-linux-py312.lock /tmp/requirements.lock
RUN python -m venv /opt/deerflow-runtime \
    && /opt/deerflow-runtime/bin/python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/requirements.lock
COPY outputs/deer-flow-v2.0/backend/packages/harness /app/outputs/deer-flow-v2.0/backend/packages/harness
COPY outputs/deer-flow/backend/packages/extension-api /app/outputs/deer-flow/backend/packages/extension-api
COPY outputs/deer-flow-v2.0/LICENSE /app/licenses/deerflow-v2.0-LICENSE
COPY outputs/deer-flow/LICENSE /app/licenses/deerflow-extension-api-LICENSE
COPY outputs/experiment-diagnostics /app/outputs/experiment-diagnostics
RUN mkdir -p /app/work/home /app/outputs/experiment-diagnostics/reports \
    && chown -R 10001:10001 /app/work /app/outputs/experiment-diagnostics/reports
ENV HOME=/app/work/home PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
RUN /opt/deerflow-runtime/bin/python /app/outputs/experiment-diagnostics/launch.py doctor
EXPOSE 8890
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD ["/opt/deerflow-runtime/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8890/health', timeout=3)"]
CMD ["/opt/deerflow-runtime/bin/python", "outputs/experiment-diagnostics/launch.py", "console", "--host", "0.0.0.0", "--port", "8890"]
