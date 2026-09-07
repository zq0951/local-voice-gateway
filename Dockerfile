FROM docker.m.daocloud.io/library/python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources || true && \
    sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list || true

RUN apt-get update && apt-get install -y \
    gcc \
    alsa-utils \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --no-deps "openwakeword>=0.6.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["python", "main.py"]
