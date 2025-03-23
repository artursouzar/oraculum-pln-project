# syntax=docker/dockerfile:1
# app/Dockerfile
FROM python:3.12-slim
LABEL authors="tiagocardoso"

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt


# Configura variáveis de ambiente
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]