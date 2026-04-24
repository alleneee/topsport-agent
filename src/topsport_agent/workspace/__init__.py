"""Per-session disk workspace: sandboxes file_ops to a bounded directory.

Why: ToolContext.workspace_root is the sandbox root honored by file_ops
(read/write/edit/list/glob/grep). Without it set, file tools can touch any
host path. SessionWorkspace binds a directory per session so HTTP-initiated
LLM tool calls can't escape to /etc/passwd etc.
"""

from .manager import SessionWorkspace, WorkspaceRegistry

__all__ = ["SessionWorkspace", "WorkspaceRegistry"]
