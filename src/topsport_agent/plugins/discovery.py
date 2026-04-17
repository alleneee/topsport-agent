"""插件发现：读取 installed_plugins.json，定位所有已安装插件的磁盘路径。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)

_DEFAULT_PLUGINS_DIR = Path.home() / ".claude" / "plugins"


@dataclass(slots=True)
class InstalledPlugin:
    """一个已安装 Claude Code 插件的元信息。"""

    name: str  # "superpowers"
    marketplace: str  # "claude-plugins-official"
    install_path: Path  # 实际磁盘路径
    version: str  # "5.0.7" 或 "unknown"


def discover_plugins(
    plugins_dir: Path | None = None,
) -> list[InstalledPlugin]:
    """读 installed_plugins.json，返回所有已安装插件。

    key 格式: "name@marketplace"。同一插件有多版本时取列表最后一项（最新）。
    跳过 install_path 不存在的条目。
    """
    base = plugins_dir or _DEFAULT_PLUGINS_DIR
    manifest_path = base / "installed_plugins.json"
    if not manifest_path.is_file():
        _logger.debug("no installed_plugins.json at %s", manifest_path)
        return []

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning("failed to read installed_plugins.json: %s", exc)
        return []

    plugins_map = data.get("plugins", {})
    result: list[InstalledPlugin] = []

    for key, entries in plugins_map.items():
        if "@" not in key or not entries:
            continue

        name, _, marketplace = key.partition("@")
        # 取最后一项（最新安装/更新的版本）
        entry = entries[-1]
        raw_path = entry.get("installPath", "")
        if not raw_path:
            continue

        install_path = Path(raw_path)
        if not install_path.is_dir():
            _logger.debug("skipping %s: path does not exist: %s", key, install_path)
            continue

        version = entry.get("version", "unknown")
        result.append(
            InstalledPlugin(
                name=name,
                marketplace=marketplace,
                install_path=install_path,
                version=version,
            )
        )

    result.sort(key=lambda p: (p.marketplace, p.name))
    return result
