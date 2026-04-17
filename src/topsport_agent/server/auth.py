"""HTTP 鉴权：Bearer token → principal 映射 + FastAPI 依赖。

设计原则 —— 分离身份与授权：
- tokens 字典把不透明 bearer 映射到 principal 名
- session_id 由 principal 做命名空间前缀，杜绝跨 principal 的 session 劫持
- compare_digest 避免字符串比较的定时侧信道

生产部署必须设置 tokens；单 token 模式用 `AUTH_TOKEN` env 即可，多租户走 JSON 文件。
默认 required=True，缺少 token 时 AuthConfig 构造抛错 —— secure by default。
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Header, HTTPException, Request

_ANONYMOUS = "anonymous"


@dataclass(slots=True, frozen=True)
class AuthConfig:
    """鉴权策略。

    tokens: 不透明 token 字符串 -> principal 名。多 token 支持多租户。
    required=False 时 require_principal 返回 "anonymous"，仅供 CLI / 单元测试使用。
    """

    required: bool = True
    tokens: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.required and not self.tokens:
            raise ValueError(
                "AuthConfig: required=True but tokens is empty; "
                "set AUTH_TOKEN env, provide a tokens file, or pass required=False"
            )
        for token, principal in self.tokens.items():
            if not token:
                raise ValueError("AuthConfig: token must be non-empty")
            if not principal:
                raise ValueError(
                    "AuthConfig: principal must be non-empty for every token"
                )

    def resolve(self, presented_token: str) -> str | None:
        """常量时间比较 presented_token 和所有已知 token；命中返回 principal，否则 None。"""
        presented = presented_token.encode()
        match: str | None = None
        # 遍历所有条目，保证即使命中也不提前返回（消除条目数带来的定时侧信道）
        for token, principal in self.tokens.items():
            if hmac.compare_digest(presented, token.encode()):
                match = principal
        return match

    @classmethod
    def disabled(cls) -> "AuthConfig":
        """显式关闭鉴权的构造入口，避免每次手写 required=False。"""
        return cls(required=False, tokens={})

    @classmethod
    def from_single_token(cls, token: str, principal: str = "default") -> "AuthConfig":
        return cls(required=True, tokens={token: principal})

    @classmethod
    def from_tokens_file(cls, path: str | Path) -> "AuthConfig":
        """JSON 文件结构: {"<token>": "<principal>", ...}。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise ValueError(
                f"auth tokens file {path}: must be a JSON object of str->str"
            )
        return cls(required=True, tokens=dict(data))


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


async def require_principal(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    """FastAPI 依赖：返回已认证的 principal 名。

    required=False 时返回 "anonymous"。required=True 时未带或非法 bearer 返回 401。
    配合 WWW-Authenticate 头让客户端知道所需 scheme。
    """
    cfg: AuthConfig = request.app.state.auth_config
    if not cfg.required:
        return _ANONYMOUS

    presented = _extract_bearer(authorization)
    if presented is None:
        raise HTTPException(
            status_code=401,
            detail="missing Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    principal = cfg.resolve(presented)
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def namespace_session_id(principal: str, user_hint: str | None) -> str:
    """把 principal 编入 session 主键，阻止跨 principal 访问他人 session。

    principal 内部用双冒号分隔保留，避免与用户自选 hint 碰撞。
    """
    hint = user_hint or ""
    return f"{principal}::{hint}" if hint else principal
