# ARG inicial para escolher o modo
ARG TARGET=cpu

#########################
# Imagem base por TARGET
#########################
FROM python:3.10-slim-bookworm AS base_cpu
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base_gpu
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip

#########################
# Estágio final
#########################
FROM base_${TARGET} AS final

WORKDIR /app

# Copiar dependências primeiro
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

EXPOSE 8000
CMD ["python", "src/main.py"]
