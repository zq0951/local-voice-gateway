#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 若存在本地 .env 配置，优先加载
if [ -f "$DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$DIR/.env" 2>/dev/null || true
    set +a
fi

# 智能探测 Python 运行时 (支持已激活环境、.venv、自定义 PYTHON_EXEC)
if [ -z "$PYTHON_EXEC" ]; then
    if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
        PIP_EXEC="$VIRTUAL_ENV/bin/pip"
    elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
        PYTHON_EXEC="$CONDA_PREFIX/bin/python"
        PIP_EXEC="$CONDA_PREFIX/bin/pip"
    elif [ -f "$DIR/.venv/bin/python" ]; then
        PYTHON_EXEC="$DIR/.venv/bin/python"
        PIP_EXEC="$DIR/.venv/bin/pip"
    elif command -v python3 &>/dev/null; then
        PYTHON_EXEC="$(command -v python3)"
        PIP_EXEC="$(command -v pip3 || command -v pip)"
    fi
fi

echo "=========================================================="
echo "📦 Local Voice Gateway 一键环境检测与依赖安装"
echo "=========================================================="

# 若未找到隔离的虚拟环境，优先在项目根目录初始化标准 .venv
if [ -z "$PYTHON_EXEC" ] || [ ! -f "$PYTHON_EXEC" ] || [ -z "$VIRTUAL_ENV$CONDA_PREFIX" ]; then
    if [ ! -f "$DIR/.venv/bin/python" ] && command -v python3 &>/dev/null; then
        echo "🛠️ 未检测到已激活的虚拟环境，正在项目内初始化标准隔离环境 (.venv)..."
        if python3 -m venv "$DIR/.venv" 2>/dev/null; then
            PYTHON_EXEC="$DIR/.venv/bin/python"
            PIP_EXEC="$DIR/.venv/bin/pip"
            echo "✅ 成功创建本地虚拟环境: $DIR/.venv"
        fi
    fi
fi

if [ -z "$PYTHON_EXEC" ] || [ ! -f "$PYTHON_EXEC" ]; then
    echo "❌ 未检测到可用的 Python 3 运行时"
    echo "💡 请先安装 Python 3.10+ 并激活虚拟环境，或在 .env 中设置 PYTHON_EXEC"
    exit 1
fi

echo "🐍 当前 Python 运行时: $($PYTHON_EXEC --version) ($PYTHON_EXEC)"

# 确保系统音频驱动依赖
OS_NAME="$(uname -s)"
if [ "$OS_NAME" = "Darwin" ]; then
    echo "🍎 检测到 macOS 运行环境 (使用 CoreAudio 驱动)"
    if command -v brew &>/dev/null; then
        if ! brew list portaudio &>/dev/null 2>&1; then
            echo "📦 通过 Homebrew 安装系统音频组件 (portaudio, ffmpeg)..."
            brew install portaudio ffmpeg 2>/dev/null || true
        fi
    else
        echo "ℹ️ 提示: 若需录音播放支持，请确保已安装 portaudio (可通过 'brew install portaudio' 安装)"
    fi
elif [ "$OS_NAME" = "Linux" ]; then
    if ! command -v arecord &>/dev/null; then
        echo "📦 安装 Linux 系统音频基础组件 (alsa-utils, portaudio)..."
        if command -v apt-get &>/dev/null; then
            if [ "$EUID" -eq 0 ]; then
                apt-get update && apt-get install -y alsa-utils libasound2-dev portaudio19-dev || true
            elif command -v sudo &>/dev/null; then
                sudo apt-get update && sudo apt-get install -y alsa-utils libasound2-dev portaudio19-dev || true
            fi
        fi
    fi
fi

# 安装 pip 依赖
echo "📦 同步 Python 依赖清单..."
if [ -n "$PIP_EXEC" ] && [ -x "$PIP_EXEC" ]; then
    PIP_CMD=("$PIP_EXEC")
else
    PIP_CMD=("$PYTHON_EXEC" -m pip)
fi

"${PIP_CMD[@]}" install -r "$DIR/requirements.txt"
echo "📦 独立安装 openwakeword (使用 --no-deps 纯 ONNX 推理模式)..."
"${PIP_CMD[@]}" install --no-deps "openwakeword>=0.6.0"

# 检查并配置模型目录
echo "🔗 检查本地模型路径..."
mkdir -p "$DIR/models/voice_profiles"
mkdir -p "$DIR/models/wakeword"

# 0. 检查 CAM++ 声纹与 OpenWakeWord 基础轻量模型 (约 30MB)
if [ ! -f "$DIR/models/campplus.onnx" ]; then
    echo "📦 自动拉取 CAM++ 声纹识别 ONNX 模型..."
    "$PYTHON_EXEC" "$DIR/utils/download_models.py" --campplus || true
fi

if [ ! -f "$DIR/models/wakeword/hey_jarvis_v0.1.onnx" ] || [ ! -f "$DIR/models/wakeword/melspectrogram.onnx" ]; then
    echo "📦 自动拉取 OpenWakeWord 唤醒词核心模型..."
    "$PYTHON_EXEC" "$DIR/utils/download_models.py" --wakeword || true
fi

# 1. 检查 FunASR 语音识别模型
if [ ! -e "$DIR/models/funasr" ]; then
    if [ -n "$FUNASR_MODEL_DIR" ] && [ -d "$FUNASR_MODEL_DIR" ]; then
        ln -sfn "$FUNASR_MODEL_DIR" "$DIR/models/funasr"
        echo "✅ 已链接外部 FunASR 模型 ($FUNASR_MODEL_DIR)"
    else
        echo "⚠️ 未在本地检测到 FunASR 语音识别模型 (models/funasr)"
        if [ -t 0 ]; then
            read -r -p "📥 是否立即从 ModelScope 镜像源极速下载 FunASR 模型 (约 1.2GB)? [Y/n] " choice
            choice="${choice:-Y}"
            if [[ "$choice" =~ ^[Yy]$ ]]; then
                "$PYTHON_EXEC" "$DIR/utils/download_models.py" --funasr
            else
                echo "ℹ️ 已跳过下载。后续可随时执行: $PYTHON_EXEC utils/download_models.py --funasr"
            fi
        else
            echo "ℹ️ 非交互环境，可通过: $PYTHON_EXEC utils/download_models.py --funasr 进行下载"
        fi
    fi
else
    echo "✅ FunASR 本地模型已就绪"
fi

# 2. 检查 MOSS-TTS 语音合成模型
if [ ! -e "$DIR/models/moss_tts" ]; then
    if [ -n "$MOSS_MODEL_DIR" ] && [ -d "$MOSS_MODEL_DIR" ]; then
        ln -sfn "$MOSS_MODEL_DIR" "$DIR/models/moss_tts"
        echo "✅ 已链接外部 MOSS-TTS 模型 ($MOSS_MODEL_DIR)"
    else
        echo "⚠️ 未在本地检测到 MOSS-TTS 语音合成模型 (models/moss_tts)"
        if [ -t 0 ]; then
            read -r -p "📥 是否立即下载 MOSS-TTS 本地离线合成模型 (约 2.5GB)? [y/N] " choice
            choice="${choice:-N}"
            if [[ "$choice" =~ ^[Yy]$ ]]; then
                "$PYTHON_EXEC" "$DIR/utils/download_models.py" --moss
            else
                echo "ℹ️ 已跳过。未安装时系统将保持轻量仿真发音；后续可执行: $PYTHON_EXEC utils/download_models.py --moss"
            fi
        else
            echo "ℹ️ 非交互环境，可通过: $PYTHON_EXEC utils/download_models.py --moss 进行下载"
        fi
    fi
else
    echo "✅ MOSS-TTS 本地模型已就绪"
fi

chmod +x "$DIR/start.sh"
echo "=========================================================="
echo "🎉 安装与检查完成！执行 ./start.sh 即可运行语音网关"
echo "=========================================================="
