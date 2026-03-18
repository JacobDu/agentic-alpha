#!/usr/bin/env python3
"""Audit factor library data completeness."""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "factor_library.db"

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# 1. 因子状态分布
print("=== 因子状态分布 ===")
cursor.execute("SELECT status, COUNT(*) FROM factors GROUP BY status")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

# 2. 因子测试结果
print("\n=== 因子测试结果 ===")
cursor.execute("SELECT COUNT(*) FROM factor_test_results")
total = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM factor_test_results WHERE significant=1")
sig = cursor.fetchone()[0]
print(f"  Total: {total}, Significant: {sig}")

# 3. 空表
print("\n=== 空表（需要填充） ===")
for table in ["factor_ic_decay", "factor_similarity", "factor_backtest_results"]:
    cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
    cnt = cursor.fetchone()[0]
    print(f"  {table}: {cnt} rows")

# 4. 因子表达式缺失
cursor.execute("SELECT COUNT(*) FROM factors WHERE expression IS NULL OR expression = ''")
no_expr = cursor.fetchone()[0]
print(f"\n=== 因子表达式缺失 ===")
print(f"  没有表达式的因子: {no_expr}")

# 5. Accepted 因子
cursor.execute("SELECT name, category FROM factors WHERE status='Accepted' ORDER BY category")
accepted = cursor.fetchall()
print(f"\n=== Accepted 因子 ({len(accepted)}) ===")
for name, cat in accepted:
    print(f"  [{cat}] {name}")

# 6. Top-30 因子
cursor.execute("""
    SELECT f.name, f.category, t.rank_icir, t.fdr_p, f.status
    FROM factors f
    JOIN factor_test_results t ON f.name = t.factor_name
    WHERE t.significant=1
    ORDER BY ABS(t.rank_icir) DESC
    LIMIT 30
""")
print(f"\n=== Top-30 因子 by |ICIR| ===")
for name, cat, icir, fdr, status in cursor.fetchall():
    w = "A" if status == "Accepted" else "B"
    print(f"  [{w}][{cat:>15}] {name:>30} ICIR={icir:+.4f} FDR={fdr:.6f}")

conn.close()
