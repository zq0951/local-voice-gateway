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

:: 智能探测 Python 运行时环境
:: 优先级:
:: 1. .env 或环境变量中的 PYTHON_EXEC
:: 2. 当前终端激活的虚拟环境 (CONDA_PREFIX / VIRTUAL_ENV)
:: 3. 项目自带的虚拟环境 (.venv)
:: 4. 系统全局 python
if not defined PYTHON_EXEC (
    if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
        set "PYTHON_EXEC=%CONDA_PREFIX%\python.exe"
    ) else if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "PYTHON_EXEC=%VIRTUAL_ENV%\Scripts\python.exe"
    ) else if exist "%DIR%\.venv\Scripts\python.exe" (
        set "PYTHON_EXEC=%DIR%\.venv\Scripts\python.exe"
    ) else (
        where python >nul 2>nul
        if !errorlevel! equ 0 (
            for /f "delims=" %%I in ('where python') do (
                if not defined PYTHON_EXEC set "PYTHON_EXEC=%%I"
            )
        )
    )
)

echo ==========================================================
echo 🚀 Local Voice Gateway 快速启动 (Windows)
echo ==========================================================

if not defined PYTHON_EXEC (
    echo ❌ 未检测到可用 Python 环境
    echo 💡 请先运行 install.bat 初始化环境，或在 .env 中配置 PYTHON_EXEC
    pause
    exit /b 1
)

if not exist "!PYTHON_EXEC!" (
    echo ❌ 指定的 Python 解释器不存在: !PYTHON_EXEC!
    pause
    exit /b 1
)

echo 🐍 选定 Python 运行时: !PYTHON_EXEC!

:: 检查 FunASR 语音识别服务连通性 (127.0.0.1:10095)
powershell -NoProfile -Command "$t = New-Object Net.Sockets.TcpClient; try { $t.Connect('127.0.0.1', 10095); Write-Host '✅ FunASR STT 语音识别服务在线 (127.0.0.1:10095)' -ForegroundColor Green } catch { Write-Host '⚠️ 提示: 127.0.0.1:10095 端口未监听，如需本地语音识别请确认 FunASR 已启动 (docker-compose up -d funasr-stt)' -ForegroundColor Yellow } finally { $t.Dispose() }" 2>nul

echo ✨ 正在启动网关主进程...
"!PYTHON_EXEC!" "%DIR%\main.py" %*

if !errorlevel! neq 0 (
    echo.
    echo ⚠️ 网关进程已退出 (退出码: !errorlevel!)
    pause
)
