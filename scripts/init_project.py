# -*- coding: utf-8 -*-
# @File    : init_project.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了项目初始化的功能，包括目录创建和环境验证。"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录（scripts/ 的上级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 需要创建的目录
REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "models",
    "utils",
    "scripts",
    "checkpoints",
    "logs",
    "notebooks",
]

# 需要创建的 __init__.py
INIT_DIRS = ["models", "utils"]

# 需要验证的Python包
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "torch": "torch",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "lightgbm": "lightgbm",
    "akshare": "akshare",
}


def create_directories() -> None:
    """创建所有缺失的项目目录。"""
    print("=" * 50)
    print("创建项目目录结构 (Project-Beta)")
    print("=" * 50)

    for dir_path in REQUIRED_DIRS:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"  [OK] {dir_path}/ (已存在)")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  [++] {dir_path}/ (已创建)")

    # 创建 __init__.py
    for dir_name in INIT_DIRS:
        init_file = PROJECT_ROOT / dir_name / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"  [++] {dir_name}/__init__.py (已创建)")


def verify_environment() -> None:
    """验证Python版本和虚拟环境。"""
    print("\n" + "=" * 50)
    print("环境验证")
    print("=" * 50)

    version = sys.version_info
    print(f"  Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor != 12:
        print(f"  [!] 建议使用Python 3.12.x，当前为 {version.major}.{version.minor}")

    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    venv_status = "[OK] 已激活" if in_venv else "[!] 未检测到虚拟环境"
    print(f"  虚拟环境: {venv_status}")
    print(f"  Python路径: {sys.executable}")


def verify_gpu() -> None:
    """验证CUDA/GPU可用性。"""
    print("\n" + "=" * 50)
    print("GPU验证")
    print("=" * 50)

    try:
        import torch
    except ImportError:
        print("  [X] PyTorch未安装，无法验证GPU")
        return

    if not torch.cuda.is_available():
        print("  [!] CUDA不可用，将使用CPU训练")
        return

    print(f"  [OK] CUDA可用")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA版本: {torch.version.cuda}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  显存: {vram_gb:.1f} GB")
    print(f"  PyTorch版本: {torch.__version__}")


def verify_packages() -> None:
    """验证必需的Python包。"""
    print("\n" + "=" * 50)
    print("依赖验证")
    print("=" * 50)

    missing: list[str] = []

    for import_name, display_name in REQUIRED_PACKAGES.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {display_name}: {version}")
        except ImportError:
            print(f"  [X] {display_name}: 未安装")
            missing.append(display_name)

    if missing:
        print(f"\n  缺失包: {', '.join(missing)}")
        print("  运行: pip install -r requirements.txt")


def check_alpha_data() -> None:
    """检查Project-Alpha数据是否可复用。"""
    print("\n" + "=" * 50)
    print("数据源检查 (Project-Alpha)")
    print("=" * 50)

    alpha_stock_data = PROJECT_ROOT.parent / "Project-Alpha" / "data" / "raw" / "stock_data.csv"
    if alpha_stock_data.exists():
        import pandas as pd
        try:
            df = pd.read_csv(alpha_stock_data, nrows=5)
            required_cols = ["open", "high", "low", "close", "volume", "amount"]
            available = [c for c in required_cols if c in df.columns]
            missing = [c for c in required_cols if c not in df.columns]
            print(f"  [OK] Project-Alpha stock_data.csv 可用")
            print(f"  可用列: {available}")
            if missing:
                print(f"  [!] 缺失列: {missing}")
            else:
                print(f"  [OK] VWAP可通过 amount/volume 计算")
        except Exception as e:
            print(f"  [X] 读取失败: {e}")
    else:
        print(f"  [X] Project-Alpha 数据未找到: {alpha_stock_data}")
        print(f"  请先运行 Project-Alpha 的 utils/data_loader.py 获取数据")


def main() -> None:
    """执行完整的项目初始化流程。"""
    print("Project-Beta 初始化")
    print()

    create_directories()
    verify_environment()
    verify_gpu()
    verify_packages()
    check_alpha_data()

    print("\n" + "=" * 50)
    print("初始化完成！")
    print("=" * 50)
    print("\n下一步:")
    print("  1. python -m utils.data_loader     # 复用Alpha数据 + 计算VWAP")
    print("  2. python -m utils.feature_engine  # 量价特征工程")
    print("  3. python -m utils.preprocess      # 预处理")
    print("  4. python train.py --model mlp     # 训练MLP")
    print("  5. python evaluate.py --model mlp  # 评估MLP")


if __name__ == "__main__":
    main()
