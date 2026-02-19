"""Clean temporary script artifacts.

Actions:
1. Remove __pycache__ directories and .pyc files under .agents/skills.
2. Optionally remove one-off scripts under ./scripts.

Usage:
    uv run python .agents/skills/qlib-env-data-prep/scripts/cleanup_temp_scripts.py
    uv run python .agents/skills/qlib-env-data-prep/scripts/cleanup_temp_scripts.py --apply
    uv run python .agents/skills/qlib-env-data-prep/scripts/cleanup_temp_scripts.py --apply --remove-root-temp
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SKILLS_DIR = PROJECT_ROOT / ".agents" / "skills"
ROOT_SCRIPTS_DIR = PROJECT_ROOT / "scripts"

ROOT_TEMP_PATTERNS = [
    re.compile(r"^tmp_.*\.py$", re.IGNORECASE),
    re.compile(r"^temp_.*\.py$", re.IGNORECASE),
    re.compile(r"^adhoc_.*\.py$", re.IGNORECASE),
    re.compile(r"^oneoff_.*\.py$", re.IGNORECASE),
    re.compile(r"^scratch_.*\.py$", re.IGNORECASE),
    re.compile(r".*\.tmp\.py$", re.IGNORECASE),
]


def _is_root_temp_script(name: str) -> bool:
    return any(p.match(name) for p in ROOT_TEMP_PATTERNS)


def _collect_targets(remove_root_temp: bool) -> tuple[list[Path], list[Path], list[Path]]:
    pycache_dirs: list[Path] = []
    pyc_files: list[Path] = []
    root_temp_scripts: list[Path] = []

    for path in SKILLS_DIR.rglob("*"):
        if path.is_dir() and path.name == "__pycache__":
            pycache_dirs.append(path)
        elif path.is_file() and path.suffix == ".pyc":
            pyc_files.append(path)

    if remove_root_temp and ROOT_SCRIPTS_DIR.exists():
        for path in ROOT_SCRIPTS_DIR.iterdir():
            if path.is_file() and _is_root_temp_script(path.name):
                root_temp_scripts.append(path)

    return pycache_dirs, pyc_files, root_temp_scripts


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean temporary script artifacts")
    parser.add_argument("--apply", action="store_true", help="Apply deletion (default is dry-run)")
    parser.add_argument("--remove-root-temp", action="store_true", help="Also remove one-off scripts under ./scripts")
    args = parser.parse_args()

    pycache_dirs, pyc_files, root_temp_scripts = _collect_targets(args.remove_root_temp)

    print("=== Cleanup Plan ===")
    print(f"skills __pycache__ dirs: {len(pycache_dirs)}")
    print(f"skills .pyc files: {len(pyc_files)}")
    print(f"root temp scripts: {len(root_temp_scripts)}")
    print()

    for path in pycache_dirs:
        print(f"[DIR] {path.relative_to(PROJECT_ROOT)}")
    for path in pyc_files:
        print(f"[PYC] {path.relative_to(PROJECT_ROOT)}")
    for path in root_temp_scripts:
        print(f"[TMP] {path.relative_to(PROJECT_ROOT)}")

    if not args.apply:
        print("\nDry-run only. Use --apply to delete.")
        return 0

    for path in pycache_dirs:
        if path.exists():
            shutil.rmtree(path)
    for path in pyc_files:
        if path.exists():
            path.unlink()
    for path in root_temp_scripts:
        if path.exists() and path.name != ".gitkeep":
            path.unlink()

    print("\nCleanup completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
