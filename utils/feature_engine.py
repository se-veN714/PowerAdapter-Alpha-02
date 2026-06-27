# -*- coding: utf-8 -*-
# @File    : feature_engine.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了量价特征工程的核心函数。

三大功能:
1. build_cross_section_features: 6字段×30天滞后展开 → 180维截面特征（MLP/GBDT输入）
2. build_sequence_data: 构建(N, 30, 6)序列数据（GRU/AGRU输入）
3. build_vwap_return_label: 构建VWAP收益率标签（T+1到T+11）
"""

import numpy as np
import pandas as pd

from config import (
    PRICE_FIELDS, EXPAND_FIELDS, STATIC_FIELDS, ALPHA_FACTOR_DIRECTION,
    SEQUENCE_LENGTH, LABEL_PERIOD, LABEL_TYPE, FILLNA_VALUE,
)


def build_cross_section_features(
    df: pd.DataFrame,
    price_fields: list[str] | None = None,
    seq_length: int = SEQUENCE_LENGTH,
    fillna_value: float = FILLNA_VALUE,
) -> pd.DataFrame:
    """为MLP/GBDT构建截面特征：每个字段展开seq_length天滞后值→ N*seq_length维。

    例如: open → open_lag0, open_lag1, ..., open_lag29
          6字段 × 30天 = 180维截面特征

    Args:
        df: 包含date, stock_code, price_fields的DataFrame。
        price_fields: 量价字段列表，默认config.PRICE_FIELDS。
        seq_length: 时序展开长度，默认30。
        fillna_value: 滞后NaN的填充值，默认0。

    Returns:
        添加了_lag{0..seq_length-1}列的DataFrame。
    """
    if price_fields is None:
        price_fields = PRICE_FIELDS

    df = df.sort_values(["stock_code", "date"]).copy()
    print(f"[截面特征] 展开 {len(price_fields)} 字段 × {seq_length} 天 = {len(price_fields) * seq_length} 维")

    for field in price_fields:
        if field not in df.columns:
            print(f"  [WARN] 字段 '{field}' 不存在，跳过")
            continue
        for lag in range(seq_length):
            df[f"{field}_lag{lag}"] = df.groupby("stock_code")[field].shift(lag)

    # 填充滞后NaN（前seq_length天的样本无足够历史数据）
    lag_cols = [c for c in df.columns if "_lag" in c]
    for col in lag_cols:
        df[col] = df[col].fillna(fillna_value)

    print(f"  [OK] 截面特征完成: {len(lag_cols)} 维")
    return df


def build_sequence_data(
    df: pd.DataFrame,
    price_fields: list[str] | None = None,
    seq_length: int = SEQUENCE_LENGTH,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """为GRU/AGRU构建序列数据。

    每只股票每个交易日 → (seq_length, n_features) 序列。
    需要前seq_length-1天历史数据，不足的样本会被跳过。

    Args:
        df: 包含date, stock_code, price_fields的DataFrame。
        price_fields: 量价字段列表，默认config.PRICE_FIELDS。
        seq_length: 序列长度，默认30。

    Returns:
        tuple: (sequences, seq_dates)
            sequences: {stock_code: ndarray of shape (n_days, seq_length, n_features)}
            seq_dates: {stock_code: ndarray of shape (n_days,)} 每条序列对应的目标日期。
    """
    if price_fields is None:
        price_fields = PRICE_FIELDS

    df = df.sort_values(["stock_code", "date"]).copy()

    # 确保所有price_fields存在
    available_fields = [f for f in price_fields if f in df.columns]
    n_features = len(available_fields)

    print(f"[序列数据] 构建 ({seq_length}, {n_features}) 格式序列")

    sequences: dict[str, np.ndarray] = {}
    seq_dates: dict[str, np.ndarray] = {}
    skipped = 0

    for code, group in df.groupby("stock_code"):
        group = group.sort_values("date")
        values = group[available_fields].values.astype(np.float32)  # (T, n_features)
        dates = group["date"].values.astype("datetime64[D]")  # (T,)

        if len(group) < seq_length:
            skipped += 1
            continue

        # 滑动窗口构建序列: 对每个交易日i，取[i-seq_length+1 : i+1]共seq_length天
        # 序列i对应的目标日期为 dates[i + seq_length - 1]
        n_days = len(group) - seq_length + 1
        seq_array = np.zeros((n_days, seq_length, n_features), dtype=np.float32)
        target_dates = np.zeros(n_days, dtype="datetime64[D]")
        for i in range(n_days):
            seq_array[i] = values[i : i + seq_length]
            target_dates[i] = dates[i + seq_length - 1]

        sequences[code] = seq_array
        seq_dates[code] = target_dates

    # 序列数据中仍可能有缺失值（如个别股票尾部数据缺失），统一填充为 0
    for code in sequences:
        sequences[code] = np.nan_to_num(sequences[code], nan=FILLNA_VALUE, posinf=FILLNA_VALUE, neginf=FILLNA_VALUE)

    print(f"  [OK] 序列数据完成: {len(sequences)} 只股票, 跳过 {skipped} 只（不足{seq_length}天）")
    return sequences, seq_dates


def build_vwap_return_label(
    df: pd.DataFrame,
    period: int = LABEL_PERIOD,
) -> pd.Series:
    """构建VWAP收益率标签：label = vwap_{t+period+1} / vwap_{t+1} - 1。

    论文定义: T+1到T+period的VWAP收益率（实际用了period天间隔）
    T日: 使用T日因子预测 T+1到T+period的收益
    label = vwap_{t+period} / vwap_{t+1} - 1  等价于 pct_change(period-1).shift(-1)

    注意: 不是close收益率, 是VWAP收益率（反映真实交易成本）

    Args:
        df: 包含date, stock_code, vwap列的DataFrame。
        period: 持有期天数，默认10。

    Returns:
        VWAP收益率标签Series，存在NaN（最后period天无未来数据）。
    """
    if "vwap" not in df.columns:
        raise ValueError("需要vwap列来计算VWAP收益率标签")

    # 按股票计算未来period天的VWAP收益率
    # pct_change(period-1) = vwap_{t+period-1}/vwap_t - 1, 需要shift再往后移一位
    # 实际上直接: label = (vwap_{t+period} / vwap_{t+1} - 1)
    # 等价于: vwap.shift(-period) / vwap.shift(-1) - 1
    vwap_future = df.groupby("stock_code")["vwap"].shift(-period)
    vwap_t1 = df.groupby("stock_code")["vwap"].shift(-1)
    label = vwap_future / vwap_t1 - 1

    return label


def build_close_return_label(
    df: pd.DataFrame,
    period: int = LABEL_PERIOD,
) -> pd.Series:
    """构建Close收益标签（路线A: 复现Alpha一致性）。
    
    与Alpha REPORT完全相同: label = close_{t+period} / close_{t+1} - 1
    
    为什么要用close而非VWAP？
    1. 基本面因子（EP/BP/ROE）反映的是日终定价共识，close是理论锚点
    2. VWAP含日内流动性噪声，不适合基本面对齐
    3. Alpha所有17轮实验均用close标签，保持一致才能复现
    
    Args:
        df: 包含date, stock_code, close列的DataFrame。
        period: 持有期天数，默认20（Alpha一致）。
    
    Returns:
        Close收益率标签Series，NaN在末尾period天。
    """
    if "close" not in df.columns:
        raise ValueError("需要close列来计算close收益率标签")
    
    close_future = df.groupby("stock_code")["close"].shift(-period)
    close_t1 = df.groupby("stock_code")["close"].shift(-1)
    label = close_future / close_t1 - 1
    
    return label


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from utils.data_loader import load_enriched_data

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 加载富化数据（Tier2+3: 50股票+30字段）
    price_path = RAW_DATA_DIR / "price_data.csv"
    if price_path.exists():
        print(f"Loading existing price data: {price_path}")
        df = pd.read_csv(price_path, parse_dates=["date"])
    else:
        df = load_enriched_data()

    # 1. 分层特征展开：
    #    路线A: EXPAND_FIELDS=[]，无滞后展开，仅保留当日因子
    #    路线B: 量价+衍生 → N天滞后展开 + 基本面仅当日值
    if EXPAND_FIELDS:
        df = build_cross_section_features(df, price_fields=EXPAND_FIELDS)

    # 保留当日因子列（路线B: STATIC_FIELDS=[]，跳过）
    static_cols = [c for c in STATIC_FIELDS if c in df.columns]
    if static_cols:
        # 路线A: 因子方向调整（负方向因子取反，确保"越大越好"）
        for col in static_cols:
            df[col] = df[col].fillna(FILLNA_VALUE)
            if col in ALPHA_FACTOR_DIRECTION and ALPHA_FACTOR_DIRECTION[col] == -1:
                df[col] = -df[col]
        print(f"  [OK] 因子: {len(static_cols)} 维 (方向已调整)")

    # 2. 收益率标签（双标签：VWAP 10天 for 路线B, Close 20天 for 路线A回退）
    print(f"  VWAP收益标签: T+1 到 T+11 (10天) [路线B: 论文设定]")
    df["label_vwap10"] = build_vwap_return_label(df, period=10)
    print(f"  Close收益标签: T+1 到 T+21 (20天) [路线A回退]")
    df["label_close20"] = build_close_return_label(df, period=20)
    # 路线B: 默认 label = label_vwap10（论文VWAP 10天标签）
    df["label"] = df["label_vwap10"]

    # 3. 序列数据（需要原始PRICE_FIELDS列，在清理之前构建）
    sequences, seq_dates = build_sequence_data(df)

    # 4. 清理截面特征列：删除原始非lag字段
    #    只保留: date, stock_code, label, label_close20, label_vwap10, _lag列, 静态因子列
    keep_cols = ["date", "stock_code", "label", "label_close20", "label_vwap10"]
    keep_cols += [c for c in df.columns if "_lag" in c]
    keep_cols += static_cols
    df = df[[c for c in keep_cols if c in df.columns]]

    total_features = len([c for c in df.columns if "_lag" in c]) + len(static_cols)
    print(f"  总截面特征: {total_features} 维 (展开{len(EXPAND_FIELDS)*SEQUENCE_LENGTH} + 静态{len(static_cols)})")

    # 保存序列数据（供GRU/AGRU使用）
    seq_path = PROCESSED_DATA_DIR / "sequences.npz"
    np.savez(seq_path, **{str(k): v for k, v in sequences.items()})
    print(f"  [OK] 序列数据已保存: {seq_path}")

    seq_dates_path = PROCESSED_DATA_DIR / "seq_dates.npz"
    np.savez(seq_dates_path, **{str(k): v for k, v in seq_dates.items()})
    print(f"  [OK] 序列日期已保存: {seq_dates_path}")

    # 保存处理后的截面数据
    output_path = PROCESSED_DATA_DIR / "processed_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n处理后的数据已保存: {output_path}")

    # 验证
    lag_cols = [c for c in df.columns if "_lag" in c]
    print(f"截面特征列数: {len(lag_cols)}")
    print(f"标签统计: {df['label'].describe()}")
    print(f"序列数据股票数: {len(sequences)}")
