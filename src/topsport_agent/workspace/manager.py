"""SessionWorkspace + WorkspaceRegistry: per-session disk sandbox roots.

Layout produced per session:

    <base>/<safe_session_id>/
        files/       # sandbox root passed to ToolContext.workspace_root

Other sub-dirs (screenshots/, images/, blobs/) can be added as other
subsystems adopt this module; the initial scope is file_ops only.

Security invariants:
  - session_id is sanitised before use as directory name (colons in
    namespaced ids would work on POSIX but are a portability footgun;
    we also prevent path-traversal via '..' components).
  - acquire() is idempotent: re-entering the same session reuses its
    files_dir rather than creating a duplicate.
  - release() defaults to preserving the workspace (ops / debug use);
    set delete=True for multi-tenant fresh-per-session policy.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)


_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]")


def _sanitise_session_id(session_id: str) -> str:
    """Map session_id to a directory-safe token.

    Current session ids in this project include ':' (namespacing like
    `principal::user_hint` or `plan_id:step_id:hash`). Those are POSIX-safe
    but fail on Windows and make shell debugging awkward; sanitise once
    and stick with it.
    """
    cleaned = _UNSAFE.sub("_", session_id)
    # Guard against all-dots (`..`, `...`) which would escape the base.
    if set(cleaned) <= {"."}:
        cleaned = "_"
    return cleaned or "_"


@dataclass(frozen=True, slots=True)
class SessionWorkspace:
    """Paths bound to one session's on-disk scratch area.

    `root` is the session's top-level directory; `files_dir` is the only
    path currently fed to `ToolContext.workspace_root` (the file_ops
    sandbox root). Additional sub-dirs can grow here as browser/image
    generation adopt the model.
    """

    session_id: str
    root: Path
    files_dir: Path


class WorkspaceRegistry:
    """Allocates and tracks SessionWorkspace instances under a base directory.

    Thread-safety: acquire() / release() are safe for concurrent use because
    mkdir(exist_ok=True) and rmtree are atomic-enough for the single-server
    case. Multi-process server deployments should set a shared-FS base +
    accept that release() may race with an active process (default is
    non-destructive so no data loss).
    """

    def __init__(self, base: Path | str) -> None:
        self._base = Path(base).expanduser()
        self._base.mkdir(parents=True, exist_ok=True)

    @property
    def base(self) -> Path:
        return self._base

    def acquire(self, session_id: str) -> SessionWorkspace:
        """Ensure the session's workspace exists and return its paths.

        Idempotent: repeated calls for the same session_id return the
        same paths and don't mutate existing files.
        """
        safe = _sanitise_session_id(session_id)
        root = self._base / safe
        files = root / "files"
        files.mkdir(parents=True, exist_ok=True)
        return SessionWorkspace(session_id=session_id, root=root, files_dir=files)

    def release(self, session_id: str, *, delete: bool = False) -> None:
        """Release a session's workspace.

        delete=False (default): keep the directory — useful for debugging,
          post-mortem inspection, and shared multi-tenant retention policies.
        delete=True: shutil.rmtree the session root. Errors are logged but
          don't propagate (release should never break session close paths).
        """
        if not delete:
            return
        safe = _sanitise_session_id(session_id)
        root = self._base / safe
        if not root.exists():
            return
        try:
            shutil.rmtree(root)
        except OSError as exc:
            _logger.warning(
                "workspace release failed session=%s root=%s: %r",
                session_id, root, exc,
            )
