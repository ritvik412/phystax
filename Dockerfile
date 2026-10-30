# Lab UI and offline benchmark in a container.
#   docker build -t phystax .
#   docker run --rm -p 127.0.0.1:8000:8000 --env-file .env phystax
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "phystax.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
