"""OpenSandbox 集成：session 级沙箱绑定 + 工具 source。

探索阶段最小可行实现：
- `OpenSandboxPool`：session_id -> sandbox 的 lazy 绑定
- `OpenSandboxToolSource`：shell / read_file / write_file 三个工具

生产接入通过 `OpenSandboxPool.from_config` 懒加载 opensandbox 包；
测试通过 `sandbox_factory` 注入 mock，无需安装 opensandbox。
"""
from .binding import SessionSandboxBinding
from .pool import OpenSandboxPool, SandboxFactory
from .tool_source import OpenSandboxToolSource

__all__ = [
    "OpenSandboxPool",
    "OpenSandboxToolSource",
    "SandboxFactory",
    "SessionSandboxBinding",
]
