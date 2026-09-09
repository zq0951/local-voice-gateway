@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"

:: 若存在 .env，尝试读取 PYTHON_EXEC
if exist "%DIR%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%DIR%\.env") do (
        if "%%A"=="PYTHON_EXEC" set "PYTHON_EXEC=%%B"
    )
)

:: 探测 Python 与 pip 运行时
if not defined PYTHON_EXEC (
    if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "PYTHON_EXEC=%VIRTUAL_ENV%\Scripts\python.exe"
        set "PIP_EXEC=%VIRTUAL_ENV%\Scripts\pip.exe"
    ) else if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON_EXEC=%CONDA_PREFIX%\python.exe"
        set "PIP_EXEC=%CONDA_PREFIX%\Scripts\pip.exe"
    ) else if exist "%DIR%\.venv\Scripts\python.exe" (
        set "PYTHON_EXEC=%DIR%\.venv\Scripts\python.exe"
        set "PIP_EXEC=%DIR%\.venv\Scripts\pip.exe"
    ) else (
        where python >nul 2>nul
        if !errorlevel! equ 0 (
            for /f "delims=" %%I in ('where python') do (
                if not defined PYTHON_EXEC set "PYTHON_EXEC=%%I"
            )
            set "PIP_EXEC=!PYTHON_EXEC! -m pip"
        )
    )
)

echo ==========================================================
echo 📦 Local Voice Gateway 一键环境检测与依赖安装 (Windows)
echo ==========================================================

:: 若未找到隔离环境，优先在项目根目录初始化标准 .venv
if not defined VIRTUAL_ENV if not defined CONDA_PREFIX (
    if not exist "%DIR%\.venv\Scripts\python.exe" (
        where python >nul 2>nul
        if !errorlevel! equ 0 (
            echo 🛠️ 未检测到已激活的虚拟环境，正在项目内初始化标准隔离环境 (.venv)...
            python -m venv "%DIR%\.venv"
            if exist "%DIR%\.venv\Scripts\python.exe" (
                set "PYTHON_EXEC=%DIR%\.venv\Scripts\python.exe"
                set "PIP_EXEC=%DIR%\.venv\Scripts\pip.exe"
                echo ✅ 成功创建本地虚拟环境: %DIR%\.venv
            )
        )
    )
)

if not defined PYTHON_EXEC (
    echo ❌ 未检测到可用的 Python 3 运行时
    echo 💡 请先安装 Python 3.10+ (安装时务必勾选 "Add Python to PATH") 并重试
    pause
    exit /b 1
)

echo 🐍 当前 Python 运行时:
"!PYTHON_EXEC!" --version

:: 安装 pip 依赖
echo.
echo 📦 正在安装/同步 Python 依赖清单 (requirements.txt)...
"!PIP_EXEC!" install -r "%DIR%\requirements.txt"

echo.
echo 📦 独立安装 openwakeword (使用 --no-deps 纯 ONNX 推理模式)...
"!PIP_EXEC!" install --no-deps "openwakeword>=0.6.0"

:: 检查并配置模型目录
echo.
echo 🔗 检查本地模型目录...
if not exist "%DIR%\models\voice_profiles" (
    mkdir "%DIR%\models\voice_profiles"
)

:: 1. 检查 FunASR 语音识别模型
if not exist "%DIR%\models\funasr" (
    echo.
    echo ⚠️ 未在本地检测到 FunASR 语音识别模型 (models\funasr)
    set /p CHOICE_ASR="📥 是否立即从 ModelScope 镜像源极速下载 FunASR 模型 (约 1.2GB)? [Y/n]: "
    if "!CHOICE_ASR!"=="" set "CHOICE_ASR=Y"
    if /i "!CHOICE_ASR!"=="Y" (
        "!PYTHON_EXEC!" "%DIR%\utils\download_models.py" --funasr
    ) else (
        echo ℹ️ 已跳过下载。后续可随时执行: !PYTHON_EXEC! utils\download_models.py --funasr
    )
) else (
    echo ✅ FunASR 本地模型已就绪
)

:: 2. 检查 MOSS-TTS 语音合成模型
if not exist "%DIR%\models\moss_tts" (
    echo.
    echo ⚠️ 未在本地检测到 MOSS-TTS 语音合成模型 (models\moss_tts)
    set /p CHOICE_MOSS="📥 是否立即下载 MOSS-TTS 本地离线合成模型 (约 2.5GB)? [y/N]: "
    if "!CHOICE_MOSS!"=="" set "CHOICE_MOSS=N"
    if /i "!CHOICE_MOSS!"=="Y" (
        "!PYTHON_EXEC!" "%DIR%\utils\download_models.py" --moss
    ) else (
        echo ℹ️ 已跳过。未安装时系统将保持轻量仿真发音；后续可执行: !PYTHON_EXEC! utils\download_models.py --moss
    )
) else (
    echo ✅ MOSS-TTS 本地模型已就绪
)

echo.
echo ==========================================================
echo 🎉 安装与环境配置完成！双击或执行 start.bat 即可运行语音网关
echo ==========================================================
pause
