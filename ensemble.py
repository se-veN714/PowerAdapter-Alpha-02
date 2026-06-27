# -*- coding: utf-8 -*-
# @File    : ensemble.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了ICIR加权Voting集成函数。

论文§2.3: 模型相关性分析与模型集成
- weight_i(t) = ICIR_i(过去60日) / sum(所有模型ICIR)
- ensemble_factor(t) = sum(weight_i(t) * factor_i(t))
"""

import numpy as np
import pandas as pd

from config import ICIR_WINDOW


def compute_rolling_icir(
    ic_series: pd.Series,
    window: int = ICIR_WINDOW,
) -> pd.Series:
    """计算滚动ICIR = mean(IC) / std(IC)。

    Args:
        ic_series: RankIC序列（以日期为索引）。
        window: 滚动窗口大小（交易日），默认60。

    Returns:
        滚动ICIR序列。
    """
    rolling_mean = ic_series.rolling(window, min_periods=window // 2).mean()
    rolling_std = ic_series.rolling(window, min_periods=window // 2).std()
    icir = rolling_mean / rolling_std.replace(0, np.nan)
    return icir.fillna(0)


def icir_weighted_voting(
    model_predictions: dict[str, np.ndarray],
    model_icirs: dict[str, float],
) -> np.ndarray:
    """ICIR加权Voting集成。

    每个模型的权重 = 该模型ICIR / sum(所有模型ICIR)
    集成因子 = sum(weight_i * pred_i)

    Args:
        model_predictions: {model_name: 预测值数组(n_samples,)}。
        model_icirs: {model_name: 当前窗口ICIR}。

    Returns:
        集成因子数组，shape=(n_samples,)。
    """
    total_icir = sum(model_icirs.values())
    if total_icir == 0:
        # 等权fallback
        weights = {name: 1.0 / len(model_icirs) for name in model_icirs}
    else:
        weights = {name: icir / total_icir for name, icir in model_icirs.items()}

    n_samples = len(next(iter(model_predictions.values())))
    ensemble = np.zeros(n_samples)

    for name, pred in model_predictions.items():
        ensemble += weights[name] * pred

    return ensemble


def build_ensemble_predictions(
    model_pred_df: dict[str, pd.DataFrame],
    ic_history: dict[str, pd.Series],
    window: int = ICIR_WINDOW,
) -> pd.Series:
    """构建逐日滚动ICIR加权集成预测。

    对每个交易日，使用截至前一天的60日ICIR计算权重。

    Args:
        model_pred_df: {model_name: DataFrame[date, stock_code, prediction]}。
        ic_history: {model_name: IC Series (日期索引)}。
        window: ICIR窗口大小。

    Returns:
        集成预测DataFrame [date, stock_code, ensemble_pred]。
    """
    all_dates = set()
    for df in model_pred_df.values():
        all_dates.update(df["date"].unique())
    all_dates = sorted(all_dates)

    ensemble_records = []

    for date in all_dates:
        # 计算当前日期的ICIR权重（用截至前一日的IC）
        model_icirs = {}
        for name, ic_series in ic_history.items():
            past_ic = ic_series[ic_series.index < date].tail(window)
            if len(past_ic) < window // 2:
                model_icirs[name] = 0.01  # fallback小值
            else:
                mean_ic = past_ic.mean()
                std_ic = past_ic.std()
                model_icirs[name] = mean_ic / std_ic if std_ic > 0 else 0.01

        # 获取当日各模型预测
        date_preds = {}
        for name, df in model_pred_df.items():
            day_data = df[df["date"] == date]
            if len(day_data) > 0:
                stocks = day_data["stock_code"].values
                preds = day_data["prediction"].values
                date_preds[name] = preds

        if len(date_preds) < 2:
            continue  # 至少要有2个模型

        # 对齐股票
        common_stocks = None
        for name, preds in date_preds.items():
            stocks = model_pred_df[name][model_pred_df[name]["date"] == date]["stock_code"].values
            if common_stocks is None:
                common_stocks = set(stocks)
            else:
                common_stocks = common_stocks & set(stocks)

        if not common_stocks:
            continue

        # ICIR加权集成
        aligned_preds = {}
        for name in date_preds:
            day_data = model_pred_df[name][model_pred_df[name]["date"] == date]
            day_data = day_data.set_index("stock_code")
            preds = day_data.loc[list(common_stocks)]["prediction"].values
            aligned_preds[name] = preds

        ensemble_pred = icir_weighted_voting(aligned_preds, model_icirs)

        # 从第一个模型DF中取出label（所有模型同日同股票label相同）
        first_model = list(model_pred_df.keys())[0]
        first_day = model_pred_df[first_model][model_pred_df[first_model]["date"] == date]
        if "label" in first_day.columns:
            label_map = dict(zip(first_day["stock_code"], first_day["label"]))
        else:
            label_map = {}

        for stock, pred in zip(sorted(common_stocks), ensemble_pred):
            ensemble_records.append({
                "date": date,
                "stock_code": stock,
                "prediction": pred,
                "label": label_map.get(stock, float("nan")),
            })

    return pd.DataFrame(ensemble_records)


if __name__ == "__main__":
    np.random.seed(42)

    # 测试ICIR加权Voting
    n_samples = 1000
    preds = {
        "MLP": np.random.randn(n_samples),
        "GBDT": np.random.randn(n_samples),
        "GRU": np.random.randn(n_samples),
    }
    icirs = {"MLP": 1.2, "GBDT": 1.5, "GRU": 0.8}

    ensemble = icir_weighted_voting(preds, icirs)
    print(f"ICIR加权集成测试:")
    print(f"  ICIRs: {icirs}")
    print(f"  Weights: MLP={1.2/3.5:.3f}, GBDT={1.5/3.5:.3f}, GRU={0.8/3.5:.3f}")
    print(f"  Ensemble shape: {ensemble.shape}")
    print(f"  Ensemble mean/std: {ensemble.mean():.4f}/{ensemble.std():.4f}")
