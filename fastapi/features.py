"""
交通事故预测系统 - 特征构造模块
包含时间特征、天气特征、路口特征的处理函数
"""

import datetime
import logging
import os
from typing import List, Dict, Any, Optional, Tuple

import httpx

from jwt_auth import init_jwt_auth, get_auth_headers
from traffic_api import get_traffic_congestion

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_time_features() -> Tuple[int, int, int, int]:
    """
    获取时间相关特征
    :return: (dayofweek, weekend, rushhour, night)
    """
    # 获取东八区当前时间
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    hour = now.hour
    dayofweek = now.isoweekday()  # 星期一=1，星期二=2，...，星期日=7
    
    # 周末判断：周六(6)或周日(7)
    weekend = 1 if dayofweek >= 6 else 0
    
    # 高峰期判断：早高峰7-9点或晚高峰17-19点
    rushhour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
    
    # 夜间判断：20-23点或0-6点
    night = 1 if (20 <= hour <= 23 or 0 <= hour <= 6) else 0
    
    logger.info(f"时间特征 - dayofweek: {dayofweek}, weekend: {weekend}, rushhour: {rushhour}, night: {night}")
    
    return dayofweek, weekend, rushhour, night


def map_weather_to_onehot(weather_text: str) -> List[int]:
    """
    将天气文本映射为one-hot编码
    :param weather_text: 天气描述文本
    :return: [sunny, cloudy, rain, fog, snow]
    """
    if not weather_text:
        # 默认晴天
        return [1, 0, 0, 0, 0]
    
    text = weather_text.strip().lower()
    
    # 初始化所有为0
    sunny = cloudy = rain = fog = snow = 0
    
    # 天气映射规则
    if "晴" in text:
        sunny = 1
    elif "多云" in text or "阴" in text:
        cloudy = 1
    elif "雨" in text:
        rain = 1
    elif "雾" in text or "霾" in text:
        fog = 1
    elif "雪" in text:
        snow = 1
    else:
        # 无法归类时默认晴天
        sunny = 1
    
    logger.info(f"天气特征映射 - 原文: '{weather_text}' -> [sunny:{sunny}, cloudy:{cloudy}, rain:{rain}, fog:{fog}, snow:{snow}]")
    
    return [sunny, cloudy, rain, fog, snow]


async def get_weather_features(lng: float, lat: float, jwt_auth_config: dict) -> Tuple[Dict[str, Any], List[int]]:
    """
    异步获取天气特征（使用JWT认证）
    :param lng: 经度
    :param lat: 纬度
    :param jwt_auth_config: JWT认证配置字典
    :return: (天气原始数据, one-hot天气特征)
    """
    # 使用JWT API HOST
    api_host = jwt_auth_config.get("api_host", "devapi.qweather.com")
    url = f"https://{api_host}/v7/weather/now"
    params = {
        "location": f"{lng},{lat}"
    }
    
    # 获取JWT认证头
    headers = get_auth_headers()
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == "200" and "now" in data:
                weather_info = data["now"]
                weather_text = weather_info.get("text", "")
                
                # 格式化返回的数据，保持与现有前端兼容
                # 确保中文字符正确编码
                location_name = data.get("basic", {}).get("location", "未知位置")
                if not location_name or location_name.strip() == "":
                    location_name = "未知位置"
                    
                formatted_weather = {
                    "locationName": location_name,
                    "temp": int(float(weather_info.get("temp", "0"))),
                    "feelsLike": int(float(weather_info.get("feelsLike", "0"))),
                    "weatherText": weather_text,
                    "humidity": int(float(weather_info.get("humidity", "0"))),
                    "vis": float(weather_info.get("vis", "0")),
                    "windSpeed": float(weather_info.get("windSpeed", "0")),
                    "windDir": weather_info.get("windDir", ""),
                    "precip": float(weather_info.get("precip", "0"))
                }
                
                weather_features = map_weather_to_onehot(weather_text)
                logger.info(f"天气API调用成功 - 坐标: ({lng}, {lat})")
                logger.info(f"天气详情 - 位置: {location_name}, 天气: {weather_text}, 温度: {weather_info.get('temp')}")
                
                return formatted_weather, weather_features
            else:
                logger.error(f"天气API返回错误: {data}")
                raise Exception(f"天气API错误: {data.get('code', 'unknown')}")
                
    except httpx.TimeoutException:
        logger.error("天气API调用超时")
        raise Exception("天气API调用超时")
    except httpx.HTTPStatusError as e:
        logger.error(f"天气API HTTP错误: {e.response.status_code} - {e.response.text}")
        raise Exception(f"天气API HTTP错误: {e.response.status_code}")
    except Exception as e:
        logger.error(f"天气API调用异常: {str(e)}")
        raise Exception(f"天气API调用异常: {str(e)}")


async def get_intersection_feature(lng: float, lat: float, amap_key: str) -> int:
    """
    异步获取路口特征
    :param lng: 经度
    :param lat: 纬度
    :param amap_key: 高德地图API密钥
    :return: is_intersection (0或1)
    """
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        "location": f"{lng},{lat}",
        "extensions": "all",
        "key": amap_key
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "1" and "regeocode" in data:
                roadinters = data["regeocode"].get("roadinters", [])
                
                if roadinters and len(roadinters) > 0:
                    # 找到距离最近的路口
                    min_distance = float('inf')
                    for intersection in roadinters:
                        distance = float(intersection.get("distance", float('inf')))
                        if distance < min_distance:
                            min_distance = distance
                    
                    # 如果最近路口距离小于30米，则认为是路口
                    is_intersection = 1 if min_distance < 30 else 0
                    logger.info(f"路口特征 - 最近路口距离: {min_distance}m, is_intersection: {is_intersection}")
                    
                    return is_intersection
                else:
                    logger.info(f"路口特征 - 无路口信息，is_intersection: 0")
                    return 0
            else:
                logger.error(f"高德逆地理编码API返回错误: {data}")
                raise Exception(f"高德API错误: {data.get('info', 'unknown')}")
                
    except httpx.TimeoutException:
        logger.error("高德API调用超时")
        raise Exception("高德API调用超时")
    except httpx.HTTPStatusError as e:
        logger.error(f"高德API HTTP错误: {e.response.status_code} - {e.response.text}")
        raise Exception(f"高德API HTTP错误: {e.response.status_code}")
    except Exception as e:
        logger.error(f"高德API调用异常: {str(e)}")
        raise Exception(f"高德API调用异常: {str(e)}")


async def construct_features(lng: float, lat: float, jwt_auth_config: str, amap_key: str) -> Tuple[List[int], Dict[str, Any]]:
    """
    构造完整的特征向量
    :param lng: 经度
    :param lat: 纬度
    :param jwt_auth_config: JWT认证配置
    :param amap_key: 高德地图API密钥
    :return: (特征向量, 额外信息)
    """
    # 1. 获取时间特征
    dayofweek, weekend, rushhour, night = get_time_features()
    
    # 2. 并行获取天气、路口和交通态势特征
    try:
        weather_task = get_weather_features(lng, lat, jwt_auth_config)
        intersection_task = get_intersection_feature(lng, lat, amap_key)
        traffic_task = get_traffic_congestion(lng, lat, amap_key)
        
        weather_info, weather_features = await weather_task
        is_intersection = await intersection_task
        traffic_info = await traffic_task
        
    except Exception as e:
        logger.error(f"外部API调用失败: {str(e)}")
        # 降级处理：使用默认值
        weather_info = {
            "locationName": "未知位置",
            "temp": 20, "feelsLike": 22, "weatherText": "晴",
            "humidity": 50, "vis": 10.0, "windSpeed": 5.0,
            "windDir": "北风", "precip": 0.0
        }
        weather_features = [1, 0, 0, 0, 0]  # 默认晴天
        is_intersection = 0
        traffic_info = {
            "congestion_score": 50,
            "congestion_level": 2,
            "congestion_name": "缓行",
            "congestion_color": "#FFFF00",
            "description": "无法获取实时路况，使用默认评估",
            "road_count": 0,
            "status_distribution": {0: 0, 1: 0, 2: 1, 3: 0, 4: 0},
            "data_source": "fallback_default"
        }
    
    # 3. 构造最终特征向量（顺序必须固定）
    feature_vector = [
        weekend,        # weekend
        rushhour,       # rushhour
        night,          # night
        weather_features[0],  # weather_sunny
        weather_features[1],  # weather_cloudy
        weather_features[2],  # weather_rain
        weather_features[3],  # weather_fog
        weather_features[4],  # weather_snow
        dayofweek,      # dayofweek
        is_intersection # is_intersection
    ]
    
    # 4. 构造返回的详细信息
    extra_info = {
        "features": {
            "weekend": weekend,
            "rushhour": rushhour,
            "night": night,
            "weather_sunny": weather_features[0],
            "weather_cloudy": weather_features[1],
            "weather_rain": weather_features[2],
            "weather_fog": weather_features[3],
            "weather_snow": weather_features[4],
            "dayofweek": dayofweek,
            "is_intersection": is_intersection
        },
        "weather_info": weather_info,
        "traffic_info": traffic_info
    }
    
    logger.info(f"特征构造完成 - 特征向量: {feature_vector}")
    
    return feature_vector, extra_info


def calculate_risk_level(risk_score: float) -> str:
    """
    根据风险分数计算风险等级
    :param risk_score: 风险分数 (0-1)
    :return: 风险等级字符串
    """
    if risk_score >= 0.8:
        return "高"
    elif risk_score >= 0.613:
        return "中"
    elif risk_score >= 0.3:
        return "低"
    else:
        return "低"