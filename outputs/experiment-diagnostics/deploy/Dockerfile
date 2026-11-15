# Build context: outputs/; this image serves the offline API only.
FROM python:3.12-slim
WORKDIR /app
COPY deer-flow/backend/packages/extension-api/deerflow_extension_api /app/deerflow_extension_api
COPY experiment-diagnostics/experiment_diagnostics /app/experiment_diagnostics
COPY experiment-diagnostics/knowledge /app/knowledge
COPY experiment-diagnostics/server.py /app/server.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8877
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8877"]
