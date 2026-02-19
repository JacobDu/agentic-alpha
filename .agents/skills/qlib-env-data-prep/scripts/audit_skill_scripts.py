"""Audit reusable scripts under .agents/skills.

Checks:
1. __pycache__/.pyc should not exist under skills.
2. Suspicious one-off script names should not exist under skill scripts dirs.
3. Hardcoded round identifiers (HEA-/SFA-/MFA-) are reported for review.

Usage:
    uv run python .agents/skills/qlib-env-data-prep/scripts/audit_skill_scripts.py
    uv run python .agents/skills/qlib-env-data-prep/scripts/audit_skill_scripts.py --strict
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SKILLS_DIR = PROJECT_ROOT / ".agents" / "skills"

SUSPICIOUS_NAME_PATTERNS = [
    re.compile(r"^tmp[_\\.-].*", re.IGNORECASE),
    re.compile(r"^temp[_\\.-].*", re.IGNORECASE),
    re.compile(r"^adhoc[_\\.-].*", re.IGNORECASE),
    re.compile(r"^oneoff[_\\.-].*", re.IGNORECASE),
    re.compile(r"^scratch[_\\.-].*", re.IGNORECASE),
    re.compile(r".*\\.tmp\\.py$", re.IGNORECASE),
]
HARDCODED_ROUND_RE = re.compile(r"\b(?:HEA|SFA|MFA)-\d{4}-\d{2}-\d{2}-\d{2}\b")


def _is_suspicious_script_name(name: str) -> bool:
    return any(p.search(name) for p in SUSPICIOUS_NAME_PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit skill scripts")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when violations are found")
    args = parser.parse_args()

    pycache_hits: list[Path] = []
    pyc_hits: list[Path] = []
    suspicious_scripts: list[Path] = []
    hardcoded_round_refs: list[tuple[Path, int, str]] = []

    for path in SKILLS_DIR.rglob("*"):
        if "__pycache__" in path.parts:
            pycache_hits.append(path)
            continue
        if path.is_file() and path.suffix == ".pyc":
            pyc_hits.append(path)
            continue

        if path.is_file() and "/scripts/" in str(path).replace("\\", "/"):
            if _is_suspicious_script_name(path.name):
                suspicious_scripts.append(path)

        if path.is_file() and path.suffix in {".py", ".md", ".sh"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if HARDCODED_ROUND_RE.search(line):
                    hardcoded_round_refs.append((path, lineno, line.strip()))

    print("=== Skill Scripts Audit ===")
    print(f"skills_dir: {SKILLS_DIR}")
    print()

    if pycache_hits:
        print(f"[FAIL] __pycache__ artifacts: {len(pycache_hits)}")
        for p in sorted(pycache_hits)[:50]:
            print(f"  - {p.relative_to(PROJECT_ROOT)}")
    else:
        print("[OK] No __pycache__ artifacts")

    if pyc_hits:
        print(f"[FAIL] .pyc artifacts: {len(pyc_hits)}")
        for p in sorted(pyc_hits)[:50]:
            print(f"  - {p.relative_to(PROJECT_ROOT)}")
    else:
        print("[OK] No .pyc artifacts")

    if suspicious_scripts:
        print(f"[FAIL] Suspicious one-off script names in skill dirs: {len(suspicious_scripts)}")
        for p in sorted(suspicious_scripts):
            print(f"  - {p.relative_to(PROJECT_ROOT)}")
    else:
        print("[OK] No suspicious one-off script names in skill dirs")

    if hardcoded_round_refs:
        print(f"[WARN] Hardcoded round references found: {len(hardcoded_round_refs)}")
        for path, lineno, line in hardcoded_round_refs[:80]:
            print(f"  - {path.relative_to(PROJECT_ROOT)}:{lineno}: {line}")
    else:
        print("[OK] No hardcoded round references")

    violations = bool(pycache_hits or pyc_hits or suspicious_scripts)
    if args.strict and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
