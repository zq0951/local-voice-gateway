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
    echo "ℹ️ 提示: 未检测到 ALSA 工具 (arecord/aplay)，网关将自动启用跨平台 PyAudio 模式托管声卡"
fi

# 检查本地 FunASR 纯离线识别模型与库
if [ -d "$DIR/models/funasr/SenseVoiceSmall" ] && "$PYTHON_EXEC" -c "import funasr" &>/dev/null; then
    echo "✅ FunASR STT 语音识别: 本地原生引擎已就绪 (纯本地私有/免 Docker)"
elif [ ! -d "$DIR/models/funasr/SenseVoiceSmall" ]; then
    echo "⚠️ 提示: 未检测到本地模型 models/funasr/SenseVoiceSmall (可执行: $PYTHON_EXEC utils/download_models.py --funasr 下载)"
else
    echo "⚠️ 提示: 未检测到 funasr 库 (请在终端执行: $PYTHON_EXEC -m pip install funasr)"
fi

echo "✨ 正在启动网关主进程: $PYTHON_EXEC"
exec "$PYTHON_EXEC" "$DIR/main.py" "$@"
