@echo off
title 交通事故风险预测系统启动器
echo.
echo ====================================
echo    交通事故风险预测系统
echo    Traffic Risk Analysis System
echo ====================================
echo.

echo [1/3] 正在启动后端服务...
cd /d "%~dp0fastapi\dist"
if not exist "backend-server.exe" (
    echo 错误：找不到后端可执行文件！
    echo 请先运行打包命令：npm run build:python
    pause
    exit /b 1
)

timeout /t 2 /nobreak >nul
start "Backend Server" cmd /k "backend-server.exe"

echo [2/3] 等待后端启动...
timeout /t 10 /nobreak >nul

echo [3/3] 正在启动前端服务...
cd /d "%~dp0frontend"
if not exist "node_modules" (
    echo 正在安装前端依赖...
    call npm install
)

start "Frontend Dev Server" cmd /k "npm run dev"

echo.
echo ====================================
echo 应用启动中...
echo 前端地址: http://localhost:5173
echo 后端地址: http://localhost:8081
echo.
echo 5秒后将自动打开浏览器...
timeout /t 5 /nobreak >nul
start http://localhost:5173

echo.
echo 按任意键退出此窗口，服务将在后台继续运行
pause >nul