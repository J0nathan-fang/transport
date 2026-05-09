"""
交通事故预测系统 - FastAPI主入口
提供天气查询和事故预测接口
"""

import logging
import os
import sys
from typing import Dict, Any
from pathlib import Path

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles


def get_resource_path(relative_path: str) -> str:
    """
    获取资源文件的绝对路径，兼容PyInstaller打包环境
    
    Args:
        relative_path: 相对路径
        
    Returns:
        str: 资源文件的绝对路径
    """
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境或正常运行时
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

from features import construct_features, calculate_risk_level
from jwt_auth import init_jwt_auth

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化JWT认证
try:
    project_id = os.getenv("QWEATHER_PROJECT_ID")
    credential_id = os.getenv("QWEATHER_CREDENTIAL_ID")
    private_key = os.getenv("QWEATHER_PRIVATE_KEY")
    
    if project_id and credential_id and private_key:
        init_jwt_auth(project_id, credential_id, private_key)
        logger.info("和风天气JWT认证初始化成功")
    else:
        logger.error("JWT认证环境变量不完整")
        
except Exception as e:
    logger.error(f"JWT认证初始化失败: {str(e)}")

# 创建FastAPI应用
app = FastAPI(
    title="交通事故风险预测API",
    description="基于机器学习的交通事故风险预测系统",
    version="1.0.0"
)

# 添加CORS中间件（开发阶段允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
dist_path = get_resource_path("dist")
if os.path.exists(dist_path):
    # 挂载assets目录，因为前端HTML中引用的是 /assets/ 路径
    assets_path = os.path.join(dist_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
    
    # 处理favicon请求，防止404错误
    @app.get("/vite.svg")
    async def favicon():
        return {"status": "favicon not found"}

# 全局异常处理中间件
@app.middleware("http")
async def global_exception_handler(request: Request, call_next):
    """全局异常处理中间件"""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"全局异常: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "内部服务器错误",
                "detail": str(e),
                "type": type(e).__name__
            }
        )

# 加载机器学习模型
try:
    logger.info("正在加载机器学习模型...")
    
    # 使用资源路径函数获取模型文件路径
    model_path = get_resource_path("models/RF_model3.pkl")
    threshold_path = get_resource_path("models/threshold.pkl")
    
    logger.info(f"模型文件路径: {model_path}")
    logger.info(f"阈值文件路径: {threshold_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"阈值文件不存在: {threshold_path}")
    
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    logger.info(f"模型加载成功 - 阈值: {threshold}")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    # 在实际生产环境中，这里应该退出应用
    model = None
    threshold = 0.613

# 定义特征顺序（必须与模型训练时一致）
FEATURE_ORDER = [
    "weekend", "rushhour", "night",
    "weather_sunny", "weather_cloudy", "weather_rain", "weather_fog", "weather_snow",
    "dayofweek", "is_intersection"
]

@app.get("/")
async def root():
    """根路径，返回前端主页"""
    index_path = get_resource_path("dist/index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        # 如果前端不存在，返回API信息
        return {
            "message": "交通事故风险预测API",
            "version": "1.0.0",
            "status": "running" if model else "model_error",
            "note": "前端文件未找到，请检查dist目录"
        }

@app.get("/api/weather")
async def get_weather(lat: float, lon: float):
    """
    天气查询接口（使用JWT认证）
    :param lat: 纬度
    :param lon: 经度
    :return: 天气信息JSON
    """
    try:
        logger.info(f"收到天气查询请求 - 坐标: ({lat}, {lon})")
        
        # 获取JWT认证配置
        jwt_config = {
            "api_host": os.getenv("QWEATHER_API_HOST", "devapi.qweather.com")
        }
        amap_key = os.getenv("AMAP_KEY", "")

        _, extra_info = await construct_features(lon, lat, jwt_config, amap_key)
        weather_info = extra_info.get("weather_info", {})
        
        logger.info(f"天气查询成功 - 位置: {weather_info.get('locationName', '未知')}")
        return weather_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"天气查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"天气查询失败: {str(e)}")

@app.post("/api/predict")
async def predict_accident_risk(request_data: Dict[str, float]):
    """
    事故预测接口（核心功能）
    :param request_data: 包含{lng, lat}的请求体
    :return: 预测结果JSON
    """
    try:
        # 检查模型是否可用
        if model is None:
            raise HTTPException(status_code=500, detail="机器学习模型未加载")
        
        # 提取经纬度
        lng = request_data.get("lng")
        lat = request_data.get("lat")
        
        if lng is None or lat is None:
            raise HTTPException(status_code=400, detail="请求参数中缺少lng或lat")
        
        logger.info(f"收到事故预测请求 - 坐标: ({lat}, {lng})")
        
        # 获取环境变量
        jwt_config = {
            "api_host": os.getenv("QWEATHER_API_HOST", "devapi.qweather.com")
        }
        amap_key = os.getenv("AMAP_KEY")
        
        if not amap_key:
            raise HTTPException(status_code=500, detail="AMAP_KEY环境变量未设置")
        
        # 构造特征向量
        feature_vector, extra_info = await construct_features(lng, lat, jwt_config, amap_key)
        
        # 验证特征向量长度
        if len(feature_vector) != 10:
            raise HTTPException(status_code=500, detail=f"特征向量长度错误: {len(feature_vector)}")
        
        # 进行预测
        risk_score = float(model.predict_proba([feature_vector])[0, 1])  # 获取正类概率
        is_accident_pred = 1 if risk_score >= threshold else 0
        risk_level = calculate_risk_level(risk_score)
        
        # 构造响应数据
        response = {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "is_accident_pred": is_accident_pred,
            "features": extra_info["features"],
            "weather_info": extra_info.get("weather_info")
        }
        
        logger.info(f"预测完成 - 风险分数: {risk_score:.4f}, 风险等级: {risk_level}, 事故预测: {is_accident_pred}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"事故预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"事故预测失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy" if model else "model_error",
        "model_loaded": model is not None,
        "threshold": threshold if model else None,
        "environment_variables": {
            "QWEATHER_PROJECT_ID": "set" if os.getenv("QWEATHER_PROJECT_ID") else "not_set",
            "QWEATHER_CREDENTIAL_ID": "set" if os.getenv("QWEATHER_CREDENTIAL_ID") else "not_set",
            "QWEATHER_API_HOST": "set" if os.getenv("QWEATHER_API_HOST") else "not_set",
            "AMAP_KEY": "set" if os.getenv("AMAP_KEY") else "not_set",
            "JWT_AUTH_INITIALIZED": bool(os.getenv("QWEATHER_PROJECT_ID") and os.getenv("QWEATHER_CREDENTIAL_ID") and os.getenv("QWEATHER_PRIVATE_KEY"))
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取端口，默认8000
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"启动FastAPI服务器，端口: {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # 开发环境启用热重载
        log_level="info"
    )