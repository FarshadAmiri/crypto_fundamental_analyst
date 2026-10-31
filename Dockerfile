FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# pypi.org direct connections are unreliable on this network (truncated responses /
# TLS resets), so resolve packages from the TUNA mirror instead. The cache mount
# keeps downloaded wheels across aborted builds.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 120 \
    -r requirements.txt

COPY app ./app
COPY web ./web

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
