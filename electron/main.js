const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const treeKill = require('tree-kill');

let mainWindow = null;
let pythonProcess = null;
const BACKEND_PORT = 8081;

function startPythonBackend() {
  return new Promise((resolve, reject) => {
    // Python后端路径（开发模式用源码，生产模式用打包后的exe）
    const isDev = !app.isPackaged;
    let pythonPath, scriptPath;
    
    if (isDev) {
      // 开发模式：直接用系统Python
      pythonPath = process.platform === 'win32' ? 'python' : 'python3';
      scriptPath = path.join(__dirname, '..', 'fastapi', 'main.py');
    } else {
      // 生产模式：使用 PyInstaller 打包后的 exe
      scriptPath = path.join(process.resourcesPath, 'backend', 'backend-server.exe');
      pythonPath = scriptPath;
    }
    
    console.log(`启动Python后端: ${pythonPath}`);
    
    // 设置环境变量
    const env = {
      ...process.env,
      PORT: String(BACKEND_PORT),
      PYTHONUNBUFFERED: '1',
    };
    
    if (isDev) {
      pythonProcess = spawn(pythonPath, [scriptPath], { env, cwd: path.join(__dirname, '..', 'fastapi') });
    } else {
      pythonProcess = spawn(pythonPath, [], { env });
    }
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(`[Python] ${data}`);
      // 检测到启动成功就 resolve
      if (data.toString().includes('Application startup complete') || 
          data.toString().includes('Uvicorn running on')) {
        console.log('Python后端启动成功！');
        resolve();
      }
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`[Python Error] ${data}`);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python进程退出，code: ${code}`);
      pythonProcess = null;
    });
    
    // 超时处理：15秒后仍未就绪也尝试连接
    setTimeout(() => {
      if (!pythonProcess.killed) {
        console.log('尝试创建UI窗口...');
        resolve();
      }
    }, 15000);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    title: '交通事故风险预测系统',
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // 开发模式加载 Vite 开发服务器，生产模式加载构建产物
  if (!app.isPackaged) {
    // 先检查前端是否运行中，如果没有则启动
    const frontendUrl = 'http://localhost:5173';
    
    setTimeout(() => {
      mainWindow.loadURL(frontendUrl).catch(() => {
        // 如果5173端口失败，尝试5174
        mainWindow.loadURL('http://localhost:5174').catch(() => {
          // 如果都失败，显示错误页面
          mainWindow.loadFile(path.join(__dirname, 'error.html'));
        });
      });
    }, 2000);
    
    // 开发模式打开DevTools
    if (process.env.NODE_ENV === 'development') {
      mainWindow.webContents.openDevTools();
    }
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'frontend', 'dist', 'index.html'));
  }
}

// 应用启动
app.whenReady().then(async () => {
  try {
    // 1. 先启动Python后端
    console.log('启动Python后端...');
    await startPythonBackend();
    
    // 2. 再创建窗口
    console.log('创建主窗口...');
    createWindow();
    
    // 3. 连接窗口关闭/打开的App事件
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
      }
    });
    
  } catch (error) {
    console.error('应用启动失败:', error);
    dialog.showErrorBox('启动错误', `应用启动失败: ${error.message}`);
    app.quit();
  }
});

// 窗口关闭处理
app.on('window-all-closed', () => {
  // 确保Python进程被终止
  if (pythonProcess && !pythonProcess.killed) {
    console.log('终止Python进程...');
    treeKill(pythonProcess.pid, 'SIGTERM', (err) => {
      if (err) {
        console.error('终止Python进程失败:', err);
      }
    });
    pythonProcess = null;
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// 应用退出前清理
app.on('before-quit', () => {
  if (pythonProcess && !pythonProcess.killed) {
    treeKill(pythonProcess.pid, 'SIGTERM');
  }
});

// 错误处理
process.on('uncaughtException', (error) => {
  console.error('未捕获的异常:', error);
  if (mainWindow) {
    dialog.showErrorBox('程序错误', `发生未捕获的异常: ${error.message}`);
  }
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('未处理的Promise拒绝:', reason);
});