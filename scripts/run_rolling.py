# -*- coding: utf-8 -*-
# @File    : run_rolling.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了批量滚动训练脚本。

使用方式:
    python scripts/run_rolling.py
"""

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 60)
    print("Project-Beta 批量滚动训练")
    print("=" * 60)

    train_script = PROJECT_ROOT / "train.py"

    if not train_script.exists():
        print(f"Error: {train_script} not found")
        return

    cmd = [
        sys.executable,
        str(train_script),
        "--mode", "rolling",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode == 0:
        print("\n滚动训练完成！")
    else:
        print(f"\n训练异常退出 (code={result.returncode})")


if __name__ == "__main__":
    main()
