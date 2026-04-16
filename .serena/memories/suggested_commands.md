# 常用命令
- 安装依赖：`uv sync`
- 运行全部测试：`uv run pytest -v`
- 运行局部测试：`uv run pytest tests/test_engine_loop.py tests/test_memory.py tests/test_mcp.py tests/test_skills.py`
- 查看仓库文件：`rg --files`
- 搜索文本：`rg "pattern" src tests README.md`
