# 交通事故风险预测系统 — Code Wiki

## 1. 项目概述

**项目名称**：交通事故风险预测系统（Traffic Risk Analysis System）

**项目定位**：基于机器学习的交通事故风险实时预测系统，整合实时天气数据、交通态势数据和地理信息，为用户提供可视化的交通事故风险评估与出行建议。

**核心能力**：
- 实时天气查询（和风天气 API + JWT 认证）
- 交通态势获取（高德地图交通态势 API）
- 路口特征识别（高德地图逆地理编码 API）
- 基于随机森林模型的交通事故风险预测
- 交互式地图可视化（高德地图 JS API）
- 桌面应用封装（Electron / pywebview）

---

## 2. 项目架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                    桌面应用层 (Desktop)                    │
│  ┌──────────────────┐    ┌───────────────────────────┐  │
│  │  Electron Shell  │    │  pywebview (desktop_app)  │  │
│  │  electron/main.js│    │  fastapi/desktop_app.py   │  │
│  └────────┬─────────┘    └────────────┬──────────────┘  │
└───────────┼───────────────────────────┼─────────────────┘
            │                           │
┌───────────┼───────────────────────────┼─────────────────┐
│           ▼        前端层 (Frontend)  ▼                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Vue 3 + Vite + Tailwind CSS                     │   │
│  │  App.vue → MapWidget / WeatherWidget /           │   │
│  │            AnalyticsWidget                        │   │
│  └──────────────────────┬───────────────────────────┘   │
└─────────────────────────┼───────────────────────────────┘
                          │ HTTP (fetch)
┌─────────────────────────┼───────────────────────────────┐
│                         ▼  后端层 (Backend)              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  FastAPI (Python) — 主后端 (端口 8081)            │   │
│  │  main.py → features.py → jwt_auth.py             │   │
│  │                        → traffic_api.py           │   │
│  │  ML模型: RF_model3.pkl + threshold.pkl           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Spring Boot (Java) — 备用后端 (端口 8081)        │   │
│  │  WeatherController → WeatherService → JwtService  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────┐
│                         ▼  外部服务层                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  和风天气 API │  │ 高德交通态势  │  │ 高德逆地理编码│  │
│  │  (JWT认证)   │  │    API       │  │    API       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 架构说明

项目采用**前后端分离 + 桌面壳**的三层架构：

| 层级 | 技术栈 | 职责 |
|------|--------|------|
| 桌面应用层 | Electron / pywebview | 进程管理、窗口管理、应用打包 |
| 前端层 | Vue 3 + Vite + Tailwind CSS | 地图交互、数据展示、用户界面 |
| 后端层 | FastAPI (Python) / Spring Boot (Java) | API 网关、ML 推理、外部 API 代理 |

> **注意**：项目存在两套后端实现。FastAPI 后端是**主用后端**，集成了完整的 ML 预测功能；Spring Boot 后端仅提供天气查询功能，属于早期版本/备用方案。

---

## 3. 目录结构

```
trea_1/
├── package.json                    # 项目根配置 (Electron 入口)
├── 启动应用.bat                     # Windows 快速启动脚本
├── README_DESKTOP.md               # 桌面应用部署指南
│
├── frontend/                       # 前端模块
│   ├── package.json                # 前端依赖配置
│   ├── vite.config.js              # Vite 构建配置
│   ├── tailwind.config.js          # Tailwind CSS 配置
│   ├── postcss.config.js           # PostCSS 配置
│   ├── index.html                  # HTML 入口
│   ├── .env                        # 开发环境变量
│   ├── .env.production             # 生产环境变量
│   ├── dist/                       # 构建产物
│   └── src/
│       ├── main.js                 # Vue 应用入口
│       ├── style.css               # 全局样式 (Tailwind 指令)
│       ├── App.vue                 # 根组件
│       └── components/
│           ├── MapWidget.vue       # 地图组件 (高德地图)
│           ├── WeatherWidget.vue   # 天气展示组件
│           └── AnalyticsWidget.vue # 风险分析组件
│
├── fastapi/                        # Python 后端模块 (主用)
│   ├── main.py                     # FastAPI 应用入口
│   ├── features.py                 # 特征构造模块
│   ├── jwt_auth.py                 # JWT 认证模块
│   ├── traffic_api.py              # 交通态势 API 模块
│   ├── desktop_app.py              # pywebview 桌面入口
│   ├── requirements.txt            # Python 依赖
│   ├── .env                        # 环境变量 (API 密钥等)
│   ├── backend-server.spec         # PyInstaller 打包配置
│   ├── models/
│   │   ├── RF_model3.pkl           # 随机森林 ML 模型
│   │   └── threshold.pkl           # 分类阈值
│   ├── dist/                       # PyInstaller 打包产物
│   │   ├── backend-server.exe      # 独立可执行文件
│   │   └── assets/                 # 前端静态资源
│   └── build/                      # PyInstaller 构建中间文件
│
├── backend/                        # Java 后端模块 (备用)
│   ├── pom.xml                     # Maven 依赖配置
│   ├── mvnw / mvnw.cmd             # Maven Wrapper
│   └── src/main/
│       ├── java/com/example/weatherdashboard/
│       │   ├── WeatherDashboardApplication.java  # Spring Boot 入口
│       │   ├── WeatherController.java            # 天气 API 控制器
│       │   ├── WebConfig.java                    # CORS 配置
│       │   ├── dto/
│       │   │   ├── WeatherDTO.java               # 天气数据传输对象
│       │   │   ├── QWeatherResponse.java         # 和风天气响应映射
│       │   │   └── Now.java                      # 天气实时数据映射
│       │   └── service/
│       │       ├── WeatherService.java           # 天气服务
│       │       └── JwtService.java               # JWT 服务
│       └── resources/
│           └── application.yml                   # Spring Boot 配置
│
└── electron/                       # Electron 桌面壳
    ├── main.js                     # Electron 主进程
    ├── error.html                  # 错误页面
    └── icon_readme.txt             # 图标说明
```

---

## 4. 主要模块详解

### 4.1 前端模块 (`frontend/`)

#### 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | ^3.4.21 | 响应式 UI 框架 |
| Vite | ^5.2.8 | 构建工具与开发服务器 |
| Tailwind CSS | ^3.4.3 | 原子化 CSS 框架 |
| @amap/amap-jsapi-loader | ^1.0.1 | 高德地图 JS API 加载器 |
| Leaflet | ^1.9.4 | 备用地图库 (已引入但未使用) |

#### 组件架构

```
App.vue (根组件)
├── Sidebar (侧边导航)
│   ├── Weather Map 菜单项
│   └── Analytics 菜单项
├── MapWidget.vue (地图组件)
│   ├── 高德地图底图
│   └── 实时路况图层 (可切换)
├── WeatherWidget.vue (天气组件, weather 模式)
│   └── 天气数据展示卡片
└── AnalyticsWidget.vue (分析组件, analytics 模式)
    ├── 综合风险评估卡片
    ├── 风险归因分析卡片
    └── 智能出行建议卡片
```

#### 关键组件说明

**App.vue** — 根组件
- 管理全局状态：`clickedCoords`（地图点击坐标）、`activeMenu`（当前菜单）
- 监听 `MapWidget` 的 `map-click` 事件，将坐标传递给右侧面板
- 根据菜单切换右侧面板显示 `WeatherWidget` 或 `AnalyticsWidget`

**MapWidget.vue** — 地图组件
- 使用 `@amap/amap-jsapi-loader` 加载高德地图 JS API 2.0
- 默认中心点：西南交通大学 (103.9870, 30.7613)
- 支持实时路况图层 (`AMap.TileLayer.Traffic`)，3 分钟自动刷新
- 点击地图时通过 `emit('map-click', coords)` 向父组件传递坐标
- 使用 `shallowRef` 存储地图实例，避免 Vue 深度代理导致的性能问题

**WeatherWidget.vue** — 天气展示组件
- 监听 `coords` prop 变化，调用 `GET /api/weather?lat=&lon=` 获取天气数据
- 展示：位置名称、温度、体感温度、天气描述、湿度、能见度、风速、风向
- 内置天气 Emoji 映射函数 `getWeatherEmoji()`

**AnalyticsWidget.vue** — 风险分析组件
- 监听 `coords` prop 变化，调用 `POST /api/predict` 获取预测结果
- 展示：综合风险评分、风险等级（低/中/高）、风险归因分析（天气/路面/交通）、智能出行建议
- 动态主题色系统：根据风险等级切换 emerald/amber/rose 配色
- 降级处理：API 调用失败时使用基于坐标的备用评估

#### 环境变量

| 变量名 | 用途 |
|--------|------|
| `VITE_AMAP_KEY` | 高德地图 Web 服务 API Key |
| `VITE_AMAP_JS_KEY` | 高德地图 JS API Key |
| `VITE_AMAP_JS_CODE` | 高德地图安全密钥 |
| `VITE_API_BASE` | 生产环境后端地址 |

---

### 4.2 FastAPI 后端模块 (`fastapi/`)

#### 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.104.1 | Web 框架 |
| Uvicorn | 0.24.0 | ASGI 服务器 |
| httpx | 0.25.2 | 异步 HTTP 客户端 |
| PyJWT | 2.8.0 | JWT Token 生成 |
| cryptography | 41.0.8 | Ed25519 密钥处理 |
| joblib | 1.3.2 | ML 模型加载 |
| numpy | 1.24.3 | 数值计算 |
| pydantic | 2.5.0 | 数据验证 |
| python-dotenv | 1.0.0 | 环境变量管理 |
| pywebview | 4.4.1 | 桌面窗口 (可选) |
| orjson | 3.9.10 | 高性能 JSON 序列化 |

#### API 接口

| 方法 | 路径 | 功能 | 参数 |
|------|------|------|------|
| GET | `/` | 返回前端主页或 API 信息 | — |
| GET | `/api/weather` | 天气查询 | `lat`, `lon` (Query) |
| POST | `/api/predict` | 事故风险预测 | `{lng, lat}` (JSON Body) |
| GET | `/health` | 健康检查 | — |

#### 模块详解

**main.py** — FastAPI 应用入口

| 函数/组件 | 说明 |
|-----------|------|
| `get_resource_path(relative_path)` | 获取资源文件绝对路径，兼容 PyInstaller 打包环境 |
| `app = FastAPI(...)` | 创建 FastAPI 应用实例 |
| `global_exception_handler` | 全局异常处理中间件，捕获未处理异常返回 500 |
| `root()` | 根路径处理器，返回前端 HTML 或 API 状态信息 |
| `get_weather(lat, lon)` | 天气查询接口，调用 `construct_features` 获取天气信息 |
| `predict_accident_risk(request_data)` | 核心预测接口，构造特征向量 → 模型推理 → 返回风险评分 |
| `health_check()` | 健康检查接口，返回模型状态和环境变量配置情况 |

**features.py** — 特征构造模块

| 函数 | 说明 | 返回值 |
|------|------|--------|
| `get_time_features()` | 获取时间相关特征 | `(dayofweek, weekend, rushhour, night)` |
| `map_weather_to_onehot(weather_text)` | 天气文本 → one-hot 编码 | `[sunny, cloudy, rain, fog, snow]` |
| `get_weather_features(lng, lat, jwt_auth_config)` | 异步获取天气特征 | `(formatted_weather, weather_features)` |
| `get_intersection_feature(lng, lat, amap_key)` | 异步获取路口特征 | `is_intersection` (0/1) |
| `construct_features(lng, lat, jwt_auth_config, amap_key)` | 构造完整特征向量 | `(feature_vector, extra_info)` |
| `calculate_risk_level(risk_score)` | 风险分数 → 风险等级 | `"高"` / `"中"` / `"低"` |

特征向量结构（10 维，顺序固定）：

| 索引 | 特征名 | 类型 | 说明 |
|------|--------|------|------|
| 0 | `weekend` | int | 是否周末 (0/1) |
| 1 | `rushhour` | int | 是否高峰期 (0/1) |
| 2 | `night` | int | 是否夜间 (0/1) |
| 3 | `weather_sunny` | int | 天气-晴 (0/1) |
| 4 | `weather_cloudy` | int | 天气-多云/阴 (0/1) |
| 5 | `weather_rain` | int | 天气-雨 (0/1) |
| 6 | `weather_fog` | int | 天气-雾/霾 (0/1) |
| 7 | `weather_snow` | int | 天气-雪 (0/1) |
| 8 | `dayofweek` | int | 星期几 (1-7) |
| 9 | `is_intersection` | int | 是否路口 (0/1) |

**jwt_auth.py** — JWT 认证模块

| 类/函数 | 说明 |
|---------|------|
| `QWeatherJWTAuth` | 和风天气 JWT 认证类 |
| `QWeatherJWTAuth.__init__(project_id, credential_id, private_key)` | 初始化，加载 Ed25519 私钥 |
| `QWeatherJWTAuth.generate_token()` | 生成 JWT Token (EdDSA 签名, 24h 有效期, 带缓存) |
| `QWeatherJWTAuth.get_auth_headers()` | 获取认证头 `{"Authorization": "Bearer <token>"}` |
| `init_jwt_auth(project_id, credential_id, private_key)` | 初始化全局 JWT 认证实例 |
| `get_jwt_auth()` | 获取全局 JWT 认证实例 |
| `get_auth_headers()` | 快捷方法：获取认证头 |

**traffic_api.py** — 交通态势模块

| 函数 | 说明 |
|------|------|
| `generate_rectangle_around_point(lng, lat, radius_km)` | 生成围绕中心点的矩形范围字符串 |
| `get_traffic_congestion(lng, lat, amap_key)` | 异步获取实时交通拥堵度 |
| `get_mock_traffic_data()` | 获取模拟交通数据 (开发测试用) |
| `get_default_traffic_data()` | 获取默认交通数据 (API 失败降级方案) |

路况级别映射：

| 状态码 | 名称 | 颜色 | 评分 |
|--------|------|------|------|
| 0 | 未知 | #999999 | 50 |
| 1 | 畅通 | #00FF00 | 20 |
| 2 | 缓行 | #FFFF00 | 50 |
| 3 | 拥堵 | #FF7800 | 75 |
| 4 | 严重拥堵 | #FF0000 | 95 |

**desktop_app.py** — pywebview 桌面入口

| 函数 | 说明 |
|------|------|
| `find_free_port(start_port, max_port)` | 查找可用端口 |
| `check_port_available(port)` | 检查端口是否可用 |
| `start_fastapi_server(port)` | 在后台线程启动 FastAPI 服务器 |
| `wait_for_server_ready(port, timeout)` | 等待服务器就绪 |
| `create_webview_window(port)` | 创建 pywebview 桌面窗口 |
| `main()` | 主入口：查找端口 → 启动后端 → 创建窗口 |

#### 环境变量

| 变量名 | 用途 | 示例 |
|--------|------|------|
| `QWEATHER_PROJECT_ID` | 和风天气项目 ID | 3M2BET8G4A |
| `QWEATHER_CREDENTIAL_ID` | JWT 凭据 ID | K8PRU294GD |
| `QWEATHER_PRIVATE_KEY` | Ed25519 私钥 (PEM 格式) | — |
| `QWEATHER_API_HOST` | 和风天气 API 地址 | pj2r6xydnw.re.qweatherapi.com |
| `AMAP_KEY` | 高德地图 Web 服务 API Key | — |
| `PORT` | 服务器端口 | 8081 |

---

### 4.3 Spring Boot 后端模块 (`backend/`)

#### 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Spring Boot | 4.0.5 | Web 框架 |
| Java | 17 | 编程语言 |
| Lombok | — | 代码简化 |
| JJWT | 0.12.5 | JWT Token 生成 |
| Jackson | — | JSON 序列化 |

#### 类结构

```
com.example.weatherdashboard
├── WeatherDashboardApplication.java   # @SpringBootApplication 入口
├── WeatherController.java             # @RestController 天气 API
├── WebConfig.java                     # @Configuration CORS 配置
├── dto/
│   ├── WeatherDTO.java                # @Data 天气数据传输对象
│   ├── QWeatherResponse.java          # @Data 和风天气响应映射
│   └── Now.java                       # 天气实时数据映射
└── service/
    ├── WeatherService.java            # @Service 天气服务
    └── JwtService.java                # @Service JWT 服务
```

#### 关键类说明

**WeatherController** — 天气 API 控制器
- `GET /api/weather?lat=&lon=` → 返回 `WeatherDTO`
- 跨域允许：`http://localhost:5173`

**WeatherService** — 天气服务
- 使用 `JwtService` 生成 JWT Token
- 通过 `RestTemplate` 调用和风天气 API
- 支持 GZIP 响应解压
- 将 `QWeatherResponse.Now` 映射为 `WeatherDTO`

**JwtService** — JWT 服务
- 从 `application.yml` 读取配置
- 使用 Ed25519 私钥签名
- Token 有效期 15 分钟

**WebConfig** — CORS 配置
- 允许 `/api/**` 路径跨域
- 允许来源：`http://localhost:5173`

#### 配置 (`application.yml`)

```yaml
server:
  port: 8081

qweather:
  project-id: "3M2BET8G4A"
  key-id: "K8PRU294GD"
  api-host: "pj2r6xydnw.re.qweatherapi.com"
  private-key: |
    -----BEGIN PRIVATE KEY-----
    ...
    -----END PRIVATE KEY-----
```

---

### 4.4 Electron 桌面壳 (`electron/`)

**main.js** — Electron 主进程

| 函数 | 说明 |
|------|------|
| `startPythonBackend()` | 启动 Python 后端进程 (开发模式用源码, 生产模式用 exe) |
| `createWindow()` | 创建 BrowserWindow (1400×900) |
| `window-all-closed` 事件 | 关闭时终止 Python 子进程 |
| `before-quit` 事件 | 退出前清理 Python 进程 |

启动流程：
1. `app.whenReady()` → 启动 Python 后端
2. 等待后端就绪 (检测 `Application startup complete` 或超时 15s)
3. 创建窗口：开发模式加载 `http://localhost:5173`，生产模式加载 `frontend/dist/index.html`

---

## 5. 数据流

### 5.1 天气查询流程

```
用户点击地图
    │
    ▼
MapWidget.vue emit('map-click', {lat, lon})
    │
    ▼
App.vue 更新 clickedCoords
    │
    ▼
WeatherWidget.vue watch(coords)
    │
    ▼
fetch GET /api/weather?lat=&lon=
    │
    ▼
FastAPI main.py → construct_features() → get_weather_features()
    │
    ▼
jwt_auth.py → generate_token() → 获取 Authorization Header
    │
    ▼
httpx → 和风天气 API (https://{api_host}/v7/weather/now)
    │
    ▼
返回天气 JSON → WeatherWidget 展示
```

### 5.2 风险预测流程

```
用户点击地图
    │
    ▼
AnalyticsWidget.vue watch(coords)
    │
    ▼
fetch POST /api/predict {lng, lat}
    │
    ▼
FastAPI main.py → predict_accident_risk()
    │
    ├─→ get_time_features()          ← 本地时间计算
    ├─→ get_weather_features()       ← 和风天气 API (JWT)
    ├─→ get_intersection_feature()   ← 高德逆地理编码 API
    └─→ get_traffic_congestion()     ← 高德交通态势 API
    │
    ▼
构造 10 维特征向量 [weekend, rushhour, night, sunny, cloudy, rain, fog, snow, dayofweek, is_intersection]
    │
    ▼
model.predict_proba([feature_vector])  ← 随机森林模型推理
    │
    ▼
risk_score ≥ threshold? → is_accident_pred
calculate_risk_level(risk_score) → "高"/"中"/"低"
    │
    ▼
返回 {risk_score, risk_level, is_accident_pred, features, weather_info}
    │
    ▼
AnalyticsWidget 展示风险评估、归因分析、出行建议
```

---

## 6. 机器学习模型

### 6.1 模型信息

| 项目 | 值 |
|------|-----|
| 模型类型 | 随机森林 (Random Forest) |
| 模型文件 | `fastapi/models/RF_model3.pkl` |
| 阈值文件 | `fastapi/models/threshold.pkl` |
| 默认阈值 | 0.613 |
| 输入维度 | 10 维特征向量 |
| 输出 | `predict_proba` 正类概率 (0~1) |

### 6.2 风险等级划分

| 风险分数范围 | 风险等级 | 前端主题色 |
|-------------|---------|-----------|
| ≥ 0.8 | 高 | rose (红色) |
| ≥ 0.613 | 中 | amber (橙色) |
| < 0.613 | 低 | emerald (绿色) |

---

## 7. 依赖关系

### 7.1 模块间依赖

```
frontend (Vue 3)
    ├── → fastapi (FastAPI) : HTTP API 调用
    │     ├── features.py → jwt_auth.py (JWT 认证)
    │     ├── features.py → traffic_api.py (交通态势)
    │     ├── features.py → 和风天气 API (外部)
    │     ├── traffic_api.py → 高德交通态势 API (外部)
    │     └── features.py → 高德逆地理编码 API (外部)
    │
    ├── → backend (Spring Boot) : HTTP API 调用 (备用)
    │     └── WeatherService → JwtService → 和风天气 API (外部)
    │
    └── electron/main.js → fastapi (子进程管理)
```

### 7.2 外部服务依赖

| 外部服务 | 用途 | 认证方式 | 配置位置 |
|----------|------|---------|---------|
| 和风天气 API | 实时天气数据 | JWT (EdDSA) | `fastapi/.env` / `application.yml` |
| 高德地图交通态势 API | 实时路况数据 | API Key | `fastapi/.env` |
| 高德地图逆地理编码 API | 路口识别 | API Key | `fastapi/.env` |
| 高德地图 JS API | 地图可视化 | JS Key + 安全密钥 | `frontend/.env` |

---

## 8. 项目运行方式

### 8.1 开发模式（推荐）

**前置条件**：
- Python 3.11+
- Node.js 18+
- Java 17+ (仅 Spring Boot 后端需要)
- Maven (仅 Spring Boot 后端需要)

**启动 FastAPI 后端**：

```bash
cd fastapi
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8081 --reload
```

或使用根目录脚本：

```bash
npm run start:backend
```

**启动前端开发服务器**：

```bash
cd frontend
npm install
npm run dev
```

或使用根目录脚本：

```bash
npm run start:frontend
```

**访问**：浏览器打开 `http://localhost:5173`

### 8.2 Electron 桌面应用模式

```bash
# 根目录安装依赖
npm install

# 开发模式启动
npm run electron:dev
# 或
npx electron .
```

### 8.3 pywebview 桌面应用模式

```bash
cd fastapi
python desktop_app.py
```

### 8.4 一键启动 (Windows)

双击 `启动应用.bat`，自动启动后端 exe + 前端开发服务器。

### 8.5 生产构建

**构建前端**：

```bash
npm run build:frontend
# 等同于 cd frontend && npm run build
```

**构建 Python 后端 (PyInstaller)**：

```bash
npm run build:python
# 等同于 cd fastapi && pyinstaller --onefile --name backend-server ...
```

**构建 Electron 安装包**：

```bash
npm run electron:build
# 先构建前端 + Python，再执行 electron-builder
```

### 8.6 Spring Boot 后端 (备用)

```bash
cd backend
./mvnw spring-boot:run
```

---

## 9. 端口配置

| 服务 | 端口 | 说明 |
|------|------|------|
| FastAPI 后端 | 8081 | 主用后端 |
| Spring Boot 后端 | 8081 | 备用后端 (不可同时运行) |
| Vite 开发服务器 | 5173 | 前端开发服务器 |
| pywebview | 8000-9000 | 自动查找可用端口 |

---

## 10. 降级与容错策略

| 场景 | 降级方案 |
|------|---------|
| 和风天气 API 不可用 | 使用默认天气数据 (晴天, 20°C) |
| 高德交通态势 API 不可用 | 使用基于时间的模拟交通数据 |
| 高德逆地理编码 API 不可用 | 默认 `is_intersection = 0` |
| ML 模型加载失败 | 预测接口返回 500 错误 |
| 前端预测 API 调用失败 | 使用基于坐标的备用评估算法 |
| JWT Token 过期 | 自动重新生成 (提前 5 分钟刷新) |
| PyInstaller 资源路径 | `get_resource_path()` 兼容打包/开发环境 |

---

## 11. 安全注意事项

- **API 密钥**：所有 API 密钥通过 `.env` 文件或 `application.yml` 管理，未硬编码在源码中
- **JWT 认证**：使用 Ed25519 非对称加密算法，私钥仅存于服务端
- **CORS**：FastAPI 开发阶段允许所有来源 (`allow_origins=["*"]`)，Spring Boot 限制为 `localhost:5173`
- **Electron 安全**：`nodeIntegration: false`, `contextIsolation: true`
- **生产环境建议**：限制 CORS 来源、使用 HTTPS、定期轮换 API 密钥
