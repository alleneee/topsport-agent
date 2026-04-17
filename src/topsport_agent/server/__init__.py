"""FastAPI HTTP + SSE 接入层。

导入该子包会拉入 fastapi/uvicorn，属于可选依赖组 `api`。
未安装 api 依赖时应避免顶层 import，请按需懒加载。
"""

from .app import create_app
from .config import ServerConfig

__all__ = ["ServerConfig", "create_app"]
