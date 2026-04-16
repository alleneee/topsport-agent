from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class BlobStore(Protocol):
    def store(self, data: str) -> str: ...

    def read(self, blob_id: str) -> str | None: ...


class FileBlobStore:
    def __init__(self, base_path: str | Path) -> None:
        self._base_path = Path(base_path)

    def store(self, data: str) -> str:
        digest = hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
        blob_id = f"blob://{digest}"
        self._base_path.mkdir(parents=True, exist_ok=True)
        path = self._base_path / f"{digest}.blob"
        path.write_text(data, encoding="utf-8")
        return blob_id

    def read(self, blob_id: str) -> str | None:
        digest = blob_id.removeprefix("blob://")
        path = self._base_path / f"{digest}.blob"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
