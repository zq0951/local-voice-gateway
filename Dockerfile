FROM docker.m.daocloud.io/library/python:3.10-slim

ENV PYTHONUNBUFFERED=1

# 替换 APT 为清华镜像源，安装底层音频驱动与 C 开发依赖（提供 libc6-dev 头文件）
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources || true && \
    sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libc6-dev \
    alsa-utils \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 升级 pip 并预装基础通用轮子，防止从 pytorch 源拉取异常 metadata 的坏包
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
    typing-extensions sympy jinja2 networkx filelock \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 CPU 专版 PyTorch (对齐 /root/smarthome 生产环境经过严格验证的 2.7.0+cpu 稳定版本)
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch==2.7.0+cpu torchaudio==2.7.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .

# 安装 openwakeword (免依赖方式避开 tflite 预编译问题) 以及其余网关依赖
RUN pip install --no-cache-dir --no-deps "openwakeword>=0.6.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --default-timeout=1000 \
    -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8765

CMD ["python", "main.py"]
