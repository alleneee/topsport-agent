from __future__ import annotations


def parse(text: str) -> tuple[dict[str, str], str]:
    """手写 frontmatter 解析器，零依赖替代 pyyaml。

    覆盖单行 key: value 和 YAML block scalar（| 保留换行，> 折叠为空格）。
    注意：key 中的连字符（如 argument-hint）原样保留，不做下划线转换。
    """
    # 标准 frontmatter 以 --- 开头和结尾，中间为 YAML 头部，之后为正文
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    header = text[4:end]
    body = text[end + 5 :].lstrip("\n")
    meta: dict[str, str] = {}
    current_key: str | None = None
    block_style: str | None = None

    for line in header.splitlines():
        # 缩进行属于前一个 block scalar 的延续内容
        if current_key is not None and block_style is not None:
            if line.startswith("  ") or line.startswith("\t"):
                existing = meta.get(current_key, "")
                stripped = line.strip()
                if block_style == "|":
                    meta[current_key] = (existing + "\n" + stripped).strip()
                else:
                    meta[current_key] = (existing + " " + stripped).strip()
                continue
            current_key = None
            block_style = None

        if ":" not in line:
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        # 值为 | 或 > 时进入 block scalar 模式，后续缩进行拼接为多行值
        if value in ("|", ">"):
            current_key = key
            block_style = value
            meta[key] = ""
            continue

        current_key = None
        block_style = None
        meta[key] = value

    return meta, body
