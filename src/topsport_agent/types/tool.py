from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from pydantic import BaseModel

# 泛型参数：让 from_model 的 handler 入参类型和 input_model 保持一致——
# 用户写 `handler: Callable[[MyInput, ToolContext], ...]` + `input_model=MyInput`
# 时，Pyright / mypy 能推断 T=MyInput 而不是退化到 BaseModel。
_BM = TypeVar("_BM", bound="BaseModel")


class TrustLevel(StrEnum):
    """工具结果信任级别，决定是否经 sanitizer 做 prompt injection 防御。

    StrEnum 的字面值兼容旧的 str 字段：`trust_level == "untrusted"` 仍然工作。
    """

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(slots=True)
class ToolContext:
    """ToolContext 随每次工具调用创建，携带会话标识和取消信号。

    cancel_event 由 Engine.cancel() 触发，长时间运行的 handler 应周期性检查。
    workspace_root 为文件类工具提供沙箱边界：None 表示 CLI 信任模式不限制；
    设置后所有路径必须 resolve 落在该根目录内，防止符号链接逃逸。
    """
    session_id: str
    call_id: str
    cancel_event: asyncio.Event
    workspace_root: Path | None = None


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]
InputValidator = Callable[[dict[str, Any]], Awaitable[str | None]]


@dataclass(slots=True)
class ToolSpec:
    """ToolSpec 是工具的完整描述：名称、JSON Schema 参数定义、异步 handler + 元数据。

    引擎每步通过 _snapshot_tools 快照当前工具列表，不跨步缓存，保证动态工具源的实时性。

    ## 元数据字段（claude-code 对标）

    **trust_level**: "trusted"（默认）对结果不做修改；"untrusted" 表示工具返回内容来自
    外部不可信源（浏览器网页、第三方 MCP 服务器等），Engine 若配了 sanitizer 会在
    结果落入 session.messages 前做注入防御处理。接受 TrustLevel 枚举或等价字符串。

    **read_only**: 声明式只读标记。True 表示该工具不产生副作用（纯查询：
    read_file / grep / list_dir / browser_get_text 等）。影响：
    - 权限决策可默认放行只读工具
    - 可被当成并发安全（若 concurrency_safe 未显式设置）

    **destructive**: 不可逆操作（删除 / 覆盖写 / 发送消息 / 关闭资源）。
    默认权限策略应对此类工具提高审批阈值。

    **concurrency_safe**: Engine 是否可以把本工具调用与同批其他 concurrency_safe 工具
    并发执行。默认 False（保守串行）；read_only 工具建议显式设 True。

    **max_result_chars**: 工具结果字符上限。None 表示使用 Engine 默认上限。超限时，
    Engine 若配了 blob_store 会自动把全量结果落盘并只把 preview + blob_ref 返给 LLM，
    防止大输出（browser_get_text 一整页、grep 大量结果等）烧 context。

    **validate_input**: 可选的 pre-flight 参数校验协程。返回 None 放行；返回 str 时
    Engine 跳过 handler，把该字符串作为错误消息直接回给 LLM 自我修正。对标 CC 的
    `validateInput() -> ValidationResult`。
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    trust_level: str = TrustLevel.TRUSTED.value
    read_only: bool = False
    destructive: bool = False
    concurrency_safe: bool = False
    max_result_chars: int | None = None
    validate_input: InputValidator | None = field(default=None)
    # 可选 pydantic 输入 schema。对标 claude-code 的 Tool.inputSchema（Zod）。
    # 设置后，from_model 会自动导出 parameters 并在 handler 外层包一层 pydantic 校验。
    # 订阅者可以通过 input_schema 做类型推断 / 文档生成 / 自动补全。
    input_schema: "type[BaseModel] | None" = field(default=None)

    @classmethod
    def from_model(
        cls,
        *,
        name: str,
        description: str,
        input_model: type[_BM],
        handler: Callable[[_BM, ToolContext], Awaitable[Any]],
        **kwargs: Any,
    ) -> "ToolSpec":
        """从 pydantic BaseModel 构造 ToolSpec：

        - `parameters` 自动由 `input_model.model_json_schema()` 生成
        - handler 入参是类型化的 BaseModel 实例（IDE 可推断字段）
        - 参数不匹配时 pydantic ValidationError 被捕获，作为 tool_result.is_error 回 LLM

        与手写 parameters+handler 的老路径共存：老工具不动，新工具走 from_model。
        """
        async def wrapped_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            from pydantic import ValidationError

            try:
                typed = input_model.model_validate(args)
            except ValidationError as exc:
                # 让 LLM 看到 pydantic 错误细节，引导自我修正
                return {
                    "error": "invalid_input",
                    "detail": exc.errors(include_url=False),
                }
            return await handler(typed, ctx)

        return cls(
            name=name,
            description=description,
            parameters=input_model.model_json_schema(),
            handler=wrapped_handler,
            input_schema=input_model,
            **kwargs,
        )
