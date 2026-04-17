"""Browser URL 访问策略：scheme 白名单 + 宿主黑名单（RFC1918 / metadata）。

默认阻断三类目标，由 LLM 驱动的 browser.navigate 不得触及：
- 非 http(s) scheme（file:// javascript: chrome:// 等）
- 环回 / 私有网段 / 链路本地 / IPv6 对应段
- 云厂商 instance metadata 的固定 IP / 域名

自定义运营可通过 BrowserURLPolicy.with_extra_denylist 追加更多黑名单；
极端受控的内网场景可传 allow_private=True 放行私有段（仍保留 metadata 拦截）。
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from urllib.parse import urlparse

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# 云厂商 instance metadata 端点（pre-DNS 字面匹配）
_METADATA_HOSTS: frozenset[str] = frozenset(
    {
        "169.254.169.254",           # AWS / GCP / Azure / Alibaba / OpenStack
        "100.100.100.200",           # Aliyun VPC
        "metadata.google.internal",  # GCP
        "metadata",                  # 裸主机名，常用于容器内覆盖
    }
)


class BrowserURLRejected(ValueError):
    """navigate 前策略拒绝；caller 应以 4xx 回显给 LLM，避免错误继续传播。"""


@dataclass(slots=True, frozen=True)
class BrowserURLPolicy:
    allowed_schemes: frozenset[str] = field(default_factory=lambda: _ALLOWED_SCHEMES)
    metadata_hosts: frozenset[str] = field(default_factory=lambda: _METADATA_HOSTS)
    extra_host_denylist: frozenset[str] = field(default_factory=frozenset)
    allow_private: bool = False  # 开放时仍拒绝 metadata / loopback

    def check(self, url: str) -> None:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in self.allowed_schemes:
            raise BrowserURLRejected(
                f"scheme {scheme!r} not allowed; permitted: {sorted(self.allowed_schemes)}"
            )

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise BrowserURLRejected(f"URL {url!r} has no hostname")

        if hostname in self.metadata_hosts:
            raise BrowserURLRejected(
                f"hostname {hostname!r} matches cloud metadata endpoint denylist"
            )
        if hostname in self.extra_host_denylist:
            raise BrowserURLRejected(
                f"hostname {hostname!r} is in operator denylist"
            )

        # IP 字面量的严格检查；域名留给 DNS 层（SSRF via DNS rebinding 仍可能，
        # 但不在本策略范围，需 egress firewall 补位）
        try:
            addr = ipaddress.ip_address(hostname)
        except ValueError:
            addr = None
        if addr is not None:
            # loopback / metadata 永远拦截，即便 allow_private=True
            if addr.is_loopback:
                raise BrowserURLRejected(f"loopback address {hostname!r} not allowed")
            if self.allow_private:
                return
            if (
                addr.is_private
                or addr.is_link_local
                or addr.is_reserved
                or addr.is_multicast
            ):
                raise BrowserURLRejected(
                    f"non-public IP {hostname!r} rejected "
                    "(private / link-local / reserved / multicast)"
                )
