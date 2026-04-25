FROM public.ecr.aws/docker/library/python:3.11.13-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

COPY requirements.txt .
RUN python -m pip install --retries 10 --timeout 100 -r requirements.txt

COPY clinicaltrial_env/ ./clinicaltrial_env/
COPY server/ ./server/
COPY protocols/ ./protocols/
COPY openenv.yaml .
COPY inference.py .
COPY demo_frontend.html .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=5)"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
