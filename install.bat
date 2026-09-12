@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
cd /d "%DIR%"

echo ==========================================================
echo 📦 Local Voice Gateway 一键环境检测与依赖安装 (Windows)
echo ==========================================================

:: 1. 若存在 .env，优先读取其中的 PYTHON_EXEC
if exist "%DIR%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%DIR%\.env") do (
        if "%%A"=="PYTHON_EXEC" set "PYTHON_EXEC=%%B"
    )
)

:: 2. 智能探测 Python 与 pip 路径 (使用平铺逻辑，彻底避开 CMD 嵌套括号解析 Bug)
if defined PYTHON_EXEC if exist "%PYTHON_EXEC%" (
    goto CHECK_PYTHON_DONE
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "PYTHON_EXEC=%VIRTUAL_ENV%\Scripts\python.exe"
    set "PIP_EXEC=%VIRTUAL_ENV%\Scripts\pip.exe"
    goto CHECK_PYTHON_DONE
)

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON_EXEC=%CONDA_PREFIX%\python.exe"
    set "PIP_EXEC=%CONDA_PREFIX%\Scripts\pip.exe"
    goto CHECK_PYTHON_DONE
)

if exist "%DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%DIR%\.venv\Scripts\python.exe"
    set "PIP_EXEC=%DIR%\.venv\Scripts\pip.exe"
    goto CHECK_PYTHON_DONE
)

:: 3. 若未检测到现有虚拟环境，探测系统 Python 并尝试初始化 .venv
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 未在系统 PATH 中检测到 python 命令
    echo 💡 请先安装 Python 3.10+ 并确保勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo 🛠️ 未检测到已激活的虚拟环境，正在项目内初始化标准隔离环境 (.venv)...
python -m venv "%DIR%\.venv"
if exist "%DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%DIR%\.venv\Scripts\python.exe"
    set "PIP_EXEC=%DIR%\.venv\Scripts\pip.exe"
    echo ✅ 成功创建本地虚拟环境: %DIR%\.venv
    goto CHECK_PYTHON_DONE
)

:: 4. 兜底使用系统全局 python
for /f "delims=" %%I in ('where python') do (
    if not defined PYTHON_EXEC set "PYTHON_EXEC=%%I"
)
set "PIP_EXEC=!PYTHON_EXEC! -m pip"

:CHECK_PYTHON_DONE
if not defined PIP_EXEC (
    set "PIP_EXEC=!PYTHON_EXEC! -m pip"
)

echo 🐍 选定 Python 运行时:
"!PYTHON_EXEC!" --version
if %errorlevel% neq 0 (
    echo ❌ Python 运行时执行失败，请检查 Python 环境
    pause
    exit /b 1
)

:: 5. 安装依赖清单
echo.
echo 📦 正在安装与同步依赖清单 (requirements.txt)...
"!PIP_EXEC!" install -r "%DIR%\requirements.txt"
if %errorlevel% neq 0 (
    echo ⚠️ 部分依赖安装出现警告，请留意上方输出
)

echo.
echo 📦 独立安装 openwakeword (纯 ONNX 推理模式)...
"!PIP_EXEC!" install --no-deps "openwakeword>=0.6.0"

:: 6. 检查并配置模型目录
echo.
echo 🔗 检查本地模型目录...
if not exist "%DIR%\models\voice_profiles" (
    mkdir "%DIR%\models\voice_profiles" 2>nul
)

:: 7. 检查 FunASR 模型
if exist "%DIR%\models\funasr" (
    echo ✅ FunASR 本地模型已就绪
    goto CHECK_MOSS
)
echo.
echo ⚠️ 未在本地检测到 FunASR 语音识别模型 (models\funasr)
set "CHOICE_ASR=Y"
set /p CHOICE_ASR="📥 是否立即从 ModelScope 镜像源极速下载 FunASR 模型 [约 1.2GB]? [Y/n]: "
if /i "%CHOICE_ASR%"=="Y" (
    "!PYTHON_EXEC!" "%DIR%\utils\download_models.py" --funasr
) else (
    echo ℹ️ 已跳过下载。后续可随时执行: python utils\download_models.py --funasr
)

:CHECK_MOSS
:: 8. 检查 MOSS-TTS 模型与运行时代码
if exist "%DIR%\models\moss_tts\infer.py" (
    echo ✅ MOSS-TTS 本地模型已就绪
    goto FINISH
)
echo.
echo ⚠️ 未在本地检测到 MOSS-TTS 语音合成模型 [models\moss_tts]
set "CHOICE_MOSS=N"
set /p CHOICE_MOSS="📥 是否立即下载 MOSS-TTS 本地离线合成模型 [约 2.5GB]? [y/N]: "
if /i "%CHOICE_MOSS%"=="Y" (
    "!PYTHON_EXEC!" "%DIR%\utils\download_models.py" --moss
) else (
    echo ℹ️ 已跳过。未安装时系统将保持轻量仿真发音；后续可执行: python utils\download_models.py --moss
)

:FINISH
echo.
echo ==========================================================
echo 🎉 安装与环境配置完成！双击或执行 start.bat 即可运行语音网关
echo ==========================================================
pause
