#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 若存在本地 .env 配置，优先加载 (支持通过 .env 自定义本地 PYTHON_EXEC，零污染公共代码)
if [ -f "$DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$DIR/.env" 2>/dev/null || true
    set +a
fi

# 智能探测 Python 运行时环境
# 探测优先级:
# 1. 环境变量或 .env 中显式指定的 PYTHON_EXEC
# 2. 当前终端已激活的虚拟环境 (CONDA_PREFIX / VIRTUAL_ENV)
# 3. 项目根目录自带的标准虚拟环境 (.venv)
# 4. 系统默认 python3
if [ -z "$PYTHON_EXEC" ]; then
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
        PYTHON_EXEC="$CONDA_PREFIX/bin/python"
    elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
    elif [ -f "$DIR/.venv/bin/python" ]; then
        PYTHON_EXEC="$DIR/.venv/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON_EXEC="$(command -v python3)"
    fi
fi

echo "=========================================================="
echo "🚀 Local Voice Gateway 快速启动"
echo "=========================================================="

if [ -z "$PYTHON_EXEC" ] || [ ! -f "$PYTHON_EXEC" ]; then
    echo "❌ 未检测到可用 Python 环境"
    echo "💡 请先运行 ./install.sh 初始化环境，或在 .env 中配置 PYTHON_EXEC"
    exit 1
fi

# 检查声卡设备
echo "🎙️ 检查音频输入输出设备..."
if ! command -v arecord &>/dev/null || ! command -v aplay &>/dev/null; then
    echo "⚠️ 警告: 未找到 arecord / aplay 工具，请安装 alsa-utils: apt-get install -y alsa-utils"
fi

# 检查 FunASR 服务连通性 (端口 10095)
if nc -z 127.0.0.1 10095 2>/dev/null || timeout 1 bash -c "</dev/tcp/127.0.0.1/10095" 2>/dev/null; then
    echo "✅ FunASR STT 语音识别服务在线 (127.0.0.1:10095)"
else
    echo "⚠️ 提示: 127.0.0.1:10095 端口未监听，如需本地语音识别请确认 FunASR 容器已启动 (docker-compose up -d funasr-stt)"
fi

echo "✨ 正在启动网关主进程: $PYTHON_EXEC"
exec "$PYTHON_EXEC" "$DIR/main.py" "$@"
