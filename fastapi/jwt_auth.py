"""
和风天气API JWT认证模块
生成JWT Token用于和风天气API认证
"""

import base64
import datetime
import hashlib
import logging
from typing import Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)


class QWeatherJWTAuth:
    """和风天气JWT认证类"""
    
    def __init__(self, project_id: str, credential_id: str, private_key: str):
        """
        初始化JWT认证
        :param project_id: 项目ID (sub)
        :param credential_id: 凭据ID (kid)  
        :param private_key: 私钥字符串 (PEM格式)
        """
        self.project_id = project_id
        self.credential_id = credential_id
        self.private_key = private_key
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[datetime.datetime] = None
        
        # 加载私钥
        self._load_private_key()
    
    def _load_private_key(self):
        """加载Ed25519私钥"""
        try:
            # 解析PEM格式的私钥
            self.private_key_obj = serialization.load_pem_private_key(
                self.private_key.encode(),
                password=None
            )
            logger.info("私钥加载成功")
        except Exception as e:
            logger.error(f"私钥加载失败: {str(e)}")
            raise
    
    def _is_token_valid(self) -> bool:
        """检查当前Token是否仍然有效"""
        if not self._cached_token or not self._token_expiry:
            return False
        
        # 提前5分钟刷新Token，避免时间差问题
        now = datetime.datetime.now(datetime.timezone.utc)
        return now < (self._token_expiry - datetime.timedelta(minutes=5))
    
    def generate_token(self) -> str:
        """
        生成JWT Token
        :return: JWT Token字符串
        """
        if self._is_token_valid():
            logger.debug("使用缓存的JWT Token")
            return self._cached_token
        
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            expires = now + datetime.timedelta(hours=24)  # 24小时有效期
            
            # JWT Header
            header = {
                "alg": "EdDSA",
                "kid": self.credential_id
            }
            
            # JWT Payload
            payload = {
                "sub": self.project_id,  # 项目ID
                "iat": int(now.timestamp()),  # 签发时间
                "exp": int(expires.timestamp())  # 过期时间
            }
            
            # 生成JWT Token
            token = jwt.encode(
                payload=payload,
                key=self.private_key_obj,
                algorithm="EdDSA",
                headers=header
            )
            
            # 缓存Token和过期时间
            self._cached_token = token
            self._token_expiry = expires
            
            logger.info(f"JWT Token生成成功，有效期至: {expires}")
            return token
            
        except Exception as e:
            logger.error(f"JWT Token生成失败: {str(e)}")
            raise
    
    def get_auth_headers(self) -> dict:
        """
        获取认证头
        :return: 包含Authorization头的字典
        """
        token = self.generate_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }


# 全局JWT认证实例
_jwt_auth: Optional[QWeatherJWTAuth] = None


def init_jwt_auth(project_id: str, credential_id: str, private_key: str):
    """
    初始化JWT认证
    :param project_id: 项目ID
    :param credential_id: 凭据ID
    :param private_key: 私钥字符串
    """
    global _jwt_auth
    _jwt_auth = QWeatherJWTAuth(project_id, credential_id, private_key)
    logger.info("JWT认证初始化完成")


def get_jwt_auth() -> Optional[QWeatherJWTAuth]:
    """获取JWT认证实例"""
    return _jwt_auth


def get_auth_headers() -> dict:
    """
    获取认证头（快捷方法）
    :return: 认证头字典
    """
    if _jwt_auth is None:
        raise RuntimeError("JWT认证未初始化，请先调用init_jwt_auth()")
    
    return _jwt_auth.get_auth_headers()