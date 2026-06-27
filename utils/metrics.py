# -*- coding: utf-8 -*-
# @File    : metrics.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了量化评估指标的函数。

包含RankIC、ICIR、IC胜率和分组回测。
"""

import numpy as np
import pandas as pd
from scipy import stats


def rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """计算Spearman秩相关系数（RankIC）。

    衡量预测排序与实际排序的单调相关性。

    Args:
        pred: 预测值数组。
        actual: 实际值数组。

    Returns:
        Spearman秩相关系数。
    """
    if len(pred) < 3:
        return 0.0
    corr, _ = stats.spearmanr(pred, actual)
    return float(corr) if not np.isnan(corr) else 0.0


def calc_ic_series(
    predictions: pd.Series,
    actuals: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    """计算每个截面的RankIC序列。

    Args:
        predictions: 预测值Series。
        actuals: 实际值Series。
        dates: 日期Series。

    Returns:
        以日期为索引的RankIC序列。
    """
    temp_df = pd.DataFrame({
        "pred": predictions.values,
        "actual": actuals.values,
        "date": dates.values,
    })

    ic_dict: dict[pd.Timestamp, float] = {}
    for date_val, group in temp_df.groupby("date"):
        ic_dict[date_val] = rank_ic(group["pred"].values, group["actual"].values)

    return pd.Series(ic_dict, name="rank_ic").sort_index()


def ic_summary(ic_series: pd.Series) -> dict[str, float]:
    """计算IC汇总指标。

    Args:
        ic_series: 截面RankIC序列。

    Returns:
        dict: rank_ic_mean, icir, ic_win_rate
    """
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std())
    icir = mean_ic / std_ic if std_ic != 0 else 0.0
    win_rate = float((ic_series > 0).mean())

    return {
        "rank_ic_mean": round(mean_ic, 6),
        "icir": round(icir, 6),
        "ic_win_rate": round(win_rate, 6),
    }


def group_return(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    return_col: str = "label",
    n_groups: int = 20,
) -> pd.DataFrame:
    """分组回测：按预测zscore分组，计算各组平均收益。

    Args:
        df: 包含预测列和收益列的DataFrame，需有date列。
        pred_col: 预测值列名。
        return_col: 实际收益列名。
        n_groups: 分组数量，默认20（论文设定）。

    Returns:
        分组回测结果DataFrame。
    """
    df = df.copy()
    df["group"] = df.groupby("date")[pred_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_groups, labels=False, duplicates="drop") + 1
    )

    result = (
        df.groupby("group")[return_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    result.columns = ["group", "mean_return", "date_count"]
    result["group"] = result["group"].astype(int)

    # 多空对冲收益
    if len(result) >= 2:
        long_short = result.iloc[-1]["mean_return"] - result.iloc[0]["mean_return"]
        ls_row = pd.DataFrame([{
            "group": "long_short",
            "mean_return": long_short,
            "date_count": result.iloc[-1]["date_count"],
        }])
        result = pd.concat([result, ls_row], ignore_index=True)

    return result


if __name__ == "__main__":
    np.random.seed(42)
    n_dates = 100
    n_stocks = 25
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    data = []
    for d in dates:
        for _ in range(n_stocks):
            data.append({
                "date": d,
                "prediction": np.random.randn(),
                "label": np.random.randn(),
            })
    test_df = pd.DataFrame(data)

    ic_series = calc_ic_series(test_df["prediction"], test_df["label"], test_df["date"])
    summary = ic_summary(ic_series)
    print(f"IC summary: {summary}")

    groups = group_return(test_df)
    print(f"\nGroup return:\n{groups}")
