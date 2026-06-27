# -*- coding: utf-8 -*-
# @File    : preprocess.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了量价数据预处理流水线。

基于SKILL: quant-cross-sectional-preprocessing
所有操作必须在groupby('date')截面内执行。

预处理步骤:
1. MAD去极值（每个滞后特征，截面内）
2. zscore标准化（截面内）
3. 缺失值填充为0
4. VWAP收益率标签构建 + 截面zscore
"""

import pandas as pd
import numpy as np

from config import MAD_MULTIPLIER, FILLNA_VALUE


def mad_clip_section(df: pd.DataFrame, col: str, n: float = MAD_MULTIPLIER) -> pd.Series:
    """对指定列做截面MAD去极值。

    MAD = Median Absolute Deviation = median(|x_i - median(x)|)
    截断范围: [median - n*MAD, median + n*MAD]

    Args:
        df: 包含date列和因子列的DataFrame。
        col: 需要处理的列名。
        n: MAD倍数，默认3。

    Returns:
        处理后的Series。
    """
    def _clip_group(group: pd.Series) -> pd.Series:
        median = group.median()
        mad = (group - median).abs().median()
        lower = median - n * mad
        upper = median + n * mad
        return group.clip(lower=lower, upper=upper)

    return df.groupby("date")[col].transform(_clip_group)


def zscore_section(df: pd.DataFrame, col: str) -> pd.Series:
    """截面zscore标准化。

    在每个交易日截面内: (x - mean) / std

    Args:
        df: 包含date列和因子列的DataFrame。
        col: 需要处理的列名。

    Returns:
        截面zscore化后的Series。
    """
    def _zscore_group(group: pd.Series) -> pd.Series:
        std = group.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return df.groupby("date")[col].transform(_zscore_group)


def preprocess_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """量价数据预处理流水线。

    对滞后特征列做MAD去极值 + zscore标准化，label做截面zscore。

    Args:
        df: 包含date, stock_code, label和_lag列的DataFrame。

    Returns:
        预处理后的DataFrame。
    """
    df = df.copy()
    print(f"Before preprocess: {len(df)} rows, {df['stock_code'].nunique()} stocks")

    # 识别所有特征列（排除所有标签列）
    _label_cols = {"label", "label_close20", "label_vwap10"}
    lag_cols = [c for c in df.columns if "_lag" in c]
    static_cols = [c for c in df.columns
                   if c not in {"date", "stock_code"} | _label_cols
                   and "_lag" not in c
                   and df[c].dtype in ("float64", "float32", "int64", "int32")]
    all_feature_cols = lag_cols + static_cols

    if not all_feature_cols:
        print("  [WARN] 无特征列，请先运行feature_engine.py")
        return df

    if lag_cols:
        print(f"  Lag features: {len(lag_cols)} columns")
    if static_cols:
        print(f"  Static factors: {len(static_cols)} columns (also standardizing)")
    if not lag_cols:
        print(f"  [Route A] Pure factor mode: {len(static_cols)} dims")

    # 跳过全NaN列
    valid_cols = [c for c in all_feature_cols if df[c].notna().sum() > 0]
    if len(valid_cols) < len(all_feature_cols):
        print(f"  [SKIP] {len(all_feature_cols) - len(valid_cols)} all-NaN columns")

    # Step 1: MAD去极值（截面内，一次性向量化）
    grouped = df.groupby("date")[valid_cols]
    medians = grouped.transform("median")
    mad = (df[valid_cols] - medians).abs().groupby(df["date"]).transform("median")
    lower = medians - MAD_MULTIPLIER * mad
    upper = medians + MAD_MULTIPLIER * mad
    df[valid_cols] = df[valid_cols].clip(lower=lower, upper=upper)
    print("[1/4] MAD clip done")

    # Step 2: zscore标准化（截面内，一次性向量化）
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    df[valid_cols] = ((df[valid_cols] - means) / stds).fillna(0)
    print("[2/4] zscore done")

    # Step 3: 缺失值填充为0（zscore后均值=0）
    df[valid_cols] = df[valid_cols].fillna(FILLNA_VALUE)
    print("[3/4] NaN fill done")

    # Step 4: Label截面zscore（双标签）
    label_cols = [c for c in ["label", "label_close20", "label_vwap10"] if c in df.columns]
    for lbl in label_cols:
        label_mean = df.groupby("date")[lbl].transform("mean")
        label_std = df.groupby("date")[lbl].transform("std").replace(0, np.nan)
        df[lbl] = ((df[lbl] - label_mean) / label_std).fillna(0)
    print(f"[4/4] Label zscore done ({len(label_cols)} labels)")

    # Step 5: 删除标签NaN的行（至少需有一个有效标签）
    n_before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].fillna(0)
    # 也填充其他标签列
    for lbl in label_cols:
        if lbl in df.columns:
            df[lbl] = df[lbl].fillna(0)
    print(f"[5/5] Cleanup: removed {n_before - len(df)} NaN-label rows")
    print(f"After preprocess: {len(df)} rows")
    return df


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from config import PROCESSED_DATA_DIR

    input_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not input_path.exists():
        print(f"请先运行 feature_engine.py 生成 {input_path}")
    else:
        df = pd.read_csv(input_path, parse_dates=["date"])
        df = preprocess_price_data(df)
        df.to_csv(input_path, index=False, encoding="utf-8-sig")
        print(f"Preprocessed data saved to {input_path}")
