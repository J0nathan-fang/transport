"""
交通态势数据获取模块
集成高德地图交通态势API，获取实时路况信息
"""

import logging
import math
from typing import Dict, List, Tuple, Any
import httpx
import os

# 从日志配置中引入logger
logger = logging.getLogger(__name__)

# 路况级别映射
TRAFFIC_STATUS_MAP = {
    0: {"name": "未知", "color": "#999999", "score": 50},
    1: {"name": "畅通", "color": "#00FF00", "score": 20},
    2: {"name": "缓行", "color": "#FFFF00", "score": 50},
    3: {"name": "拥堵", "color": "#FF7800", "score": 75},
    4: {"name": "严重拥堵", "color": "#FF0000", "score": 95}
}

def generate_rectangle_around_point(lng: float, lat: float, radius_km: float = 1.0) -> str:
    """
    生成围绕中心点的矩形范围
    :param lng: 中心点经度
    :param lat: 中心点纬度  
    :param radius_km: 半径(公里)，最大不超过5公里
    :return: 矩形范围字符串 "左下经度,左下纬度;右上经度,右上纬度"
    """
    if radius_km > 5.0:
        radius_km = 5.0
        logger.warning("矩形半径过大，调整为5公里")
    
    # 计算经纬度偏移量
    lat_diff = radius_km / 111.0  # 1度纬度约111公里
    lon_diff = radius_km / (111.0 * math.cos(math.radians(lat)))  # 经度随纬度变化
    
    min_lon = lng - lon_diff
    min_lat = lat - lat_diff
    max_lon = lng + lon_diff
    max_lat = lat + lat_diff
    
    rectangle = f"{min_lon:.6f},{min_lat:.6f};{max_lon:.6f},{max_lat:.6f}"
    return rectangle

async def get_traffic_congestion(lng: float, lat: float, amap_key: str) -> Dict[str, Any]:
    """
    获取指定位置的实时交通拥堵度
    :param lng: 经度
    :param lat: 纬度
    :param amap_key: 高德地图API密钥
    :return: 包含拥堵信息的字典
    """
    if not amap_key or amap_key.strip() == "":
        logger.warning("高德地图API密钥未配置，使用模拟交通数据")
        return get_mock_traffic_data()
    
    # 生成1公里半径的矩形范围
    rectangle = generate_rectangle_around_point(lng, lat, radius_km=1.0)
    
    url = "https://restapi.amap.com/v3/traffic/status/rectangle"
    params = {
        "key": amap_key,
        "rectangle": rectangle,
        "extensions": "all",  # 获取详细信息
        "level": 6  # 道路等级，6表示获取所有道路信息
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "1" and "trafficinfo" in data:
                traffic_info = data["trafficinfo"]
                roads = traffic_info.get("evaluation", {}).get("roads", [])
                
                if not roads:
                    logger.warning(f"指定区域({lng}, {lat})内无道路信息，返回默认值")
                    return get_default_traffic_data()
                
                # 分析路况数据
                congestion_scores = []
                status_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # 各状态数量统计
                
                total_length = 0
                for road in roads:
                    status = int(road.get("status", 0))
                    length = float(road.get("length", 0))
                    
                    status_counts[status] += 1
                    congestion_scores.append(TRAFFIC_STATUS_MAP[status]["score"] * length)
                    total_length += length
                
                # 计算加权平均拥堵度
                if total_length > 0:
                    weighted_score = sum(congestion_scores) / total_length
                else:
                    weighted_score = 50  # 默认中等拥堵
                
                # 找出主要路况状态
                main_status = max(status_counts.keys(), key=lambda k: status_counts[k])
                
                result = {
                    "congestion_score": round(weighted_score, 1),
                    "congestion_level": main_status,
                    "congestion_name": TRAFFIC_STATUS_MAP[main_status]["name"],
                    "congestion_color": TRAFFIC_STATUS_MAP[main_status]["color"],
                    "description": f"区域路况主要为{TRAFFIC_STATUS_MAP[main_status]['name']}",
                    "road_count": len(roads),
                    "status_distribution": status_counts,
                    "data_source": "amap_traffic_api"
                }
                
                logger.info(f"交通态势查询成功 - 坐标: ({lng}, {lat}), 拥堵度: {weighted_score:.1f}%, 状态: {result['congestion_name']}")
                return result
                
            else:
                error_info = data.get("info", "未知错误")
                logger.error(f"高德交通态势API错误: {error_info}")
                return get_default_traffic_data()
        
    except httpx.TimeoutException:
        logger.error("高德交通态势API调用超时")
        return get_default_traffic_data()
    except httpx.HTTPStatusError as e:
        logger.error(f"高德交通态势API HTTP错误: {e.response.status_code} - {e.response.text}")
        return get_default_traffic_data()
    except Exception as e:
        logger.error(f"高德交通态势API调用异常: {str(e)}")
        return get_default_traffic_data()

def get_mock_traffic_data() -> Dict[str, Any]:
    """
    获取模拟的交通数据（用于开发测试）
    :return: 模拟交通数据字典
    """
    import random
    from datetime import datetime
    
    # 模拟不同时间的交通模式
    hour = datetime.now().hour
    
    if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰期
        base_score = random.randint(65, 85)
        main_status = random.choice([2, 3, 4])
    elif hour >= 22 or hour <= 6:  # 夜间
        base_score = random.randint(10, 30)
        main_status = random.choice([1, 1, 2])
    else:  # 平峰期
        base_score = random.randint(30, 55)
        main_status = random.choice([1, 2, 2, 3])
    
    return {
        "congestion_score": base_score,
        "congestion_level": main_status,
        "congestion_name": TRAFFIC_STATUS_MAP[main_status]["name"],
        "congestion_color": TRAFFIC_STATUS_MAP[main_status]["color"],
        "description": f"模拟路况：{TRAFFIC_STATUS_MAP[main_status]['name']}（基于时间模式估算）",
        "road_count": random.randint(5, 15),
        "status_distribution": {0: 0, 1: 3, 2: 2, 3: 1, 4: 0},
        "data_source": "mock_time_based"
    }

def get_default_traffic_data() -> Dict[str, Any]:
    """
   获取默认交通数据（API失败时的降级方案）
   :return: 默认交通数据字典
   """
    return {
        "congestion_score": 50,
        "congestion_level": 2,
        "congestion_name": TRAFFIC_STATUS_MAP[2]["name"],
        "congestion_color": TRAFFIC_STATUS_MAP[2]["color"],
        "description": "无法获取实时路况，使用默认评估",
        "road_count": 0,
        "status_distribution": {0: 0, 1: 0, 2: 1, 3: 0, 4: 0},
        "data_source": "fallback_default"
    }