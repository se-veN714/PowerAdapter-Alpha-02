# -*- coding: utf-8 -*-
# @File    : data_loader.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了量价数据加载功能。

Tier2+3优化: 复用Project-Alpha的stock_data_with_factors.csv（50只股票+17个因子），
计算VWAP和衍生特征，构建丰富的特征体系。
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    ALPHA_FACTOR_DATA, ALPHA_STOCK_DATA,
    STOCK_POOL, PRICE_FIELDS, DERIVED_FIELDS, FACTOR_FIELDS,
    ALL_CROSS_SECTION_FIELDS, DATE_RANGE, RAW_DATA_DIR,
)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """从成交额和成交量计算VWAP近似值。

    VWAP = amount / volume
    若无法从数据源直接获取VWAP，用此方法近似。

    Args:
        df: 包含amount和volume列的DataFrame。

    Returns:
        VWAP Series，NaN占位当amount或volume无效。
    """
    if "amount" not in df.columns or "volume" not in df.columns:
        raise ValueError("需要amount和volume列来计算VWAP")
    vwap = df["amount"] / df["volume"].replace(0, float("nan"))
    return vwap


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tier2: 从原始量价数据计算衍生特征。

    所有计算按股票分组，防止跨股票数据泄露。

    Args:
        df: 包含 [date, stock_code, open, high, low, close, volume] 的DataFrame。

    Returns:
        添加了衍生特征列的DataFrame。
    """
    df = df.sort_values(["stock_code", "date"]).copy()

    # 日收益率
    df["return_1d"] = df.groupby("stock_code")["close"].pct_change(1)

    # 5日/20日收益率
    df["return_5d"] = df.groupby("stock_code")["close"].pct_change(5)
    df["return_20d"] = df.groupby("stock_code")["close"].pct_change(20)

    # 10日波动率（日收益率的标准差）
    ret = df.groupby("stock_code")["close"].pct_change()
    df["volatility_10d"] = ret.groupby(df["stock_code"]).transform(
        lambda x: x.rolling(10, min_periods=5).std()
    )

    # 量比: volume / 5日均量
    df["volume_ratio_5d"] = (
        df["volume"]
        / df.groupby("stock_code")["volume"].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
    )

    # 日内振幅: (high - low) / close
    df["intraday_spread"] = (df["high"] - df["low"]) / df["close"]

    # 隔夜跳空: close / open
    df["close_open_ratio"] = df["close"] / df["open"]

    return df


def load_enriched_data(
    stock_pool: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Tier2+3: 从Project-Alpha加载富化数据。

    数据源: stock_data_with_factors.csv（50只股票+17个因子+OHLCV）
    额外计算: VWAP + 7个衍生特征
    输出: 包含全部6+7+17=30个原始字段的DataFrame，供后续滞后展开。

    Args:
        stock_pool: 股票代码列表，默认config.STOCK_POOL。
        save: 是否保存到data/raw/。

    Returns:
        包含全部特征字段的DataFrame。
    """
    if stock_pool is None:
        stock_pool = STOCK_POOL

    # 优先使用因子数据，回退到基础数据
    data_path = ALPHA_FACTOR_DATA if ALPHA_FACTOR_DATA.exists() else ALPHA_STOCK_DATA
    if not data_path.exists():
        raise FileNotFoundError(f"数据不存在: {data_path}")

    print(f"Loading enriched data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"  Raw shape: {df.shape}, stocks: {df['stock_code'].nunique()}")

    # 过滤股票池
    df = df[df["stock_code"].astype(str).str.zfill(6).isin(stock_pool)].copy()
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    print(f"  After stock filter: {df.shape}, stocks: {df['stock_code'].nunique()}")

    # 计算VWAP
    if "vwap" not in df.columns and "amount" in df.columns and "volume" in df.columns:
        df["vwap"] = compute_vwap(df)
        print(f"  VWAP computed")
    elif "vwap" not in df.columns:
        print("  [WARN] 无法计算VWAP: 缺少amount或volume列")

    # Tier2: 计算衍生特征
    df = compute_derived_features(df)
    deriv_count = sum(1 for c in DERIVED_FIELDS if c in df.columns)
    print(f"  Derived features: {deriv_count}/{len(DERIVED_FIELDS)} computed")

    # 检查因子字段
    factor_count = sum(1 for c in FACTOR_FIELDS if c in df.columns)
    print(f"  Factor fields: {factor_count}/{len(FACTOR_FIELDS)} available")

    # 确定保留的列: 所有需要展开的字段 + date/stock_code
    available_cs_fields = [c for c in ALL_CROSS_SECTION_FIELDS if c in df.columns]
    keep_cols = ["date", "stock_code"] + available_cs_fields
    missing = set(ALL_CROSS_SECTION_FIELDS) - set(available_cs_fields)
    if missing:
        print(f"  [WARN] 缺失字段: {sorted(missing)}")

    df = df[keep_cols].copy()
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    print(f"  Final shape: {df.shape}")
    print(f"  Feature fields: {len(available_cs_fields)} (×30 lags = {len(available_cs_fields)*30} dims)")
    print(f"  Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  Stocks: {df['stock_code'].nunique()}")

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RAW_DATA_DIR / "price_data.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  Saved to: {output_path}")

    return df


def load_alpha_data(
    data_path: Path | None = None,
    stock_pool: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """向后兼容的别名，内部调用 load_enriched_data。

    Args:
        data_path: 已废弃（自动选择最优数据源）。
        stock_pool: 股票代码列表。
        save: 是否保存。

    Returns:
        包含全部特征字段的DataFrame。
    """
    return load_enriched_data(stock_pool=stock_pool, save=save)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("Project-Beta Enriched Data Loader (Tier2+3)")
    print("=" * 60)
    df = load_enriched_data()
    print(f"\nPreview:\n{df.head(10)}")
    print(f"\nColumns ({len(df.columns)}): {df.columns.tolist()}")
