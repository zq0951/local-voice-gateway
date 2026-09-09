@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "DIR=%~dp0"
if "%DIR:~-1%"=="\" set "DIR=%DIR:~0,-1%"
cd /d "%DIR%"

echo ==========================================================
echo Local Voice Gateway 快速启动 (Windows)
echo ==========================================================

:: 1. 若存在 .env，尝试读取 PYTHON_EXEC
if exist "%DIR%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%DIR%\.env") do (
        if "%%A"=="PYTHON_EXEC" set "PYTHON_EXEC=%%B"
    )
)

:: 2. 智能探测 Python 运行时
if defined PYTHON_EXEC if exist "%PYTHON_EXEC%" (
    goto FOUND_PYTHON
)

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON_EXEC=%CONDA_PREFIX%\python.exe"
    goto FOUND_PYTHON
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "PYTHON_EXEC=%VIRTUAL_ENV%\Scripts\python.exe"
    goto FOUND_PYTHON
)

if exist "%DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%DIR%\.venv\Scripts\python.exe"
    goto FOUND_PYTHON
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%I in ('where python') do (
        if not defined PYTHON_EXEC set "PYTHON_EXEC=%%I"
    )
    goto FOUND_PYTHON
)

echo [ERROR] 未检测到可用 Python 环境
echo [HINT] 请先双击运行 install.bat 初始化环境，或在 .env 中配置 PYTHON_EXEC
pause
exit /b 1

:FOUND_PYTHON
echo [Python] 选定运行时: !PYTHON_EXEC!

:: 3. 检查 FunASR 语音识别服务连通性 (纯 Python 探测，毫秒级且无 PowerShell 兼容性风险)
"!PYTHON_EXEC!" -c "import socket; s = socket.socket(); s.settimeout(0.3); res = s.connect_ex(('127.0.0.1', 10095)); print('[FunASR] 本地语音识别服务在线 (127.0.0.1:10095)' if res == 0 else '[FunASR] 提示: 10095 端口未监听 (若需识别请启动 FunASR)'); s.close()" 2>nul


echo [Gateway] 正在启动网关主进程...
"!PYTHON_EXEC!" "%DIR%\main.py" %*

if %errorlevel% neq 0 (
    echo.
    echo [Gateway] 网关进程已退出 (退出码: %errorlevel%)
    pause
)
