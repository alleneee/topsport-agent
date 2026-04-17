"""插件发现测试：单元测试 + 真实 ~/.claude/plugins 集成测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from topsport_agent.plugins.discovery import discover_plugins

# ---------------------------------------------------------------------------
# 单元测试：临时目录
# ---------------------------------------------------------------------------


def test_discover_empty_dir(tmp_path: Path) -> None:
    """没有 installed_plugins.json 时返回空列表。"""
    result = discover_plugins(tmp_path)
    assert result == []


def test_discover_malformed_json(tmp_path: Path) -> None:
    """JSON 损坏时返回空列表，不抛异常。"""
    (tmp_path / "installed_plugins.json").write_text("not json", encoding="utf-8")
    result = discover_plugins(tmp_path)
    assert result == []


def test_discover_empty_plugins(tmp_path: Path) -> None:
    """plugins 字段为空 dict 时返回空列表。"""
    (tmp_path / "installed_plugins.json").write_text(
        json.dumps({"version": 2, "plugins": {}}), encoding="utf-8"
    )
    result = discover_plugins(tmp_path)
    assert result == []


def test_discover_skips_missing_path(tmp_path: Path) -> None:
    """installPath 指向不存在的目录时跳过。"""
    data = {
        "version": 2,
        "plugins": {
            "ghost@test-marketplace": [
                {
                    "scope": "user",
                    "installPath": str(tmp_path / "nonexistent"),
                    "version": "1.0.0",
                }
            ]
        },
    }
    (tmp_path / "installed_plugins.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    result = discover_plugins(tmp_path)
    assert result == []


def test_discover_single_plugin(tmp_path: Path) -> None:
    """正常发现单个插件。"""
    plugin_dir = tmp_path / "cache" / "test-mp" / "my-plugin" / "1.0.0"
    plugin_dir.mkdir(parents=True)

    data = {
        "version": 2,
        "plugins": {
            "my-plugin@test-mp": [
                {
                    "scope": "user",
                    "installPath": str(plugin_dir),
                    "version": "1.0.0",
                }
            ]
        },
    }
    (tmp_path / "installed_plugins.json").write_text(
        json.dumps(data), encoding="utf-8"
    )

    result = discover_plugins(tmp_path)
    assert len(result) == 1
    assert result[0].name == "my-plugin"
    assert result[0].marketplace == "test-mp"
    assert result[0].install_path == plugin_dir
    assert result[0].version == "1.0.0"


def test_discover_takes_latest_version(tmp_path: Path) -> None:
    """同一插件多版本时取最后一项。"""
    old_dir = tmp_path / "cache" / "mp" / "p" / "1.0"
    new_dir = tmp_path / "cache" / "mp" / "p" / "2.0"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)

    data = {
        "version": 2,
        "plugins": {
            "p@mp": [
                {"installPath": str(old_dir), "version": "1.0"},
                {"installPath": str(new_dir), "version": "2.0"},
            ]
        },
    }
    (tmp_path / "installed_plugins.json").write_text(
        json.dumps(data), encoding="utf-8"
    )

    result = discover_plugins(tmp_path)
    assert len(result) == 1
    assert result[0].version == "2.0"
    assert result[0].install_path == new_dir


def test_discover_skips_bad_key_format(tmp_path: Path) -> None:
    """key 不含 @ 时跳过。"""
    plugin_dir = tmp_path / "cache" / "bad"
    plugin_dir.mkdir(parents=True)

    data = {
        "version": 2,
        "plugins": {
            "no-at-sign": [
                {"installPath": str(plugin_dir), "version": "1.0"}
            ]
        },
    }
    (tmp_path / "installed_plugins.json").write_text(
        json.dumps(data), encoding="utf-8"
    )

    result = discover_plugins(tmp_path)
    assert result == []


def test_discover_multiple_plugins_sorted(tmp_path: Path) -> None:
    """多个插件按 (marketplace, name) 排序。"""
    dir_b = tmp_path / "cache" / "mp-b" / "beta" / "1.0"
    dir_a = tmp_path / "cache" / "mp-a" / "alpha" / "1.0"
    dir_b.mkdir(parents=True)
    dir_a.mkdir(parents=True)

    data = {
        "version": 2,
        "plugins": {
            "beta@mp-b": [{"installPath": str(dir_b), "version": "1.0"}],
            "alpha@mp-a": [{"installPath": str(dir_a), "version": "1.0"}],
        },
    }
    (tmp_path / "installed_plugins.json").write_text(
        json.dumps(data), encoding="utf-8"
    )

    result = discover_plugins(tmp_path)
    assert len(result) == 2
    assert result[0].name == "alpha"
    assert result[0].marketplace == "mp-a"
    assert result[1].name == "beta"
    assert result[1].marketplace == "mp-b"


# ---------------------------------------------------------------------------
# 集成测试：真实 ~/.claude/plugins
# ---------------------------------------------------------------------------

_REAL_PLUGINS_DIR = Path.home() / ".claude" / "plugins"


@pytest.mark.skipif(
    not (_REAL_PLUGINS_DIR / "installed_plugins.json").is_file(),
    reason="~/.claude/plugins/installed_plugins.json not found",
)
class TestRealPlugins:
    def test_discovers_real_plugins(self) -> None:
        """能从真实 installed_plugins.json 发现至少一个插件。"""
        result = discover_plugins(_REAL_PLUGINS_DIR)
        assert len(result) > 0

    def test_all_paths_exist(self) -> None:
        """所有返回的插件 install_path 都是实际存在的目录。"""
        result = discover_plugins(_REAL_PLUGINS_DIR)
        for plugin in result:
            assert plugin.install_path.is_dir(), f"{plugin.name}: {plugin.install_path}"

    def test_superpowers_discovered(self) -> None:
        """superpowers 是常见已安装插件，应能发现。"""
        result = discover_plugins(_REAL_PLUGINS_DIR)
        names = {p.name for p in result}
        assert "superpowers" in names
