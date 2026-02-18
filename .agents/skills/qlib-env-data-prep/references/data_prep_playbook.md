# 数据准备流程手册

1. 运行 `uv run python .agents/skills/qlib-env-data-prep/scripts/prepare_data.py`。
2. 运行 `uv run python .agents/skills/qlib-env-data-prep/scripts/download_financial_data.py --phase 1`。
3. 运行 `uv run python .agents/skills/qlib-env-data-prep/scripts/download_financial_data.py --phase 2`。
4. 运行 `uv run python .agents/skills/qlib-env-data-prep/scripts/check_data.py`。
5. 运行 `uv run python .agents/skills/qlib-env-data-prep/scripts/verify_all.py`。
