# -*- coding: utf-8 -*-
# @File    : evaluate.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了模型评估入口。

评估指标（论文§2.2）:
- RankIC均值: >0.03有效
- ICIR: RankIC均值/标准差, >0.5稳定
- IC胜率: IC>0的截面占比, >50%有效
- 20组分组回测: 按预测zscore分20组，计算各组平均收益

使用方式:
    python evaluate.py                    # 评估最近训练的模型
    python evaluate.py --model mlp        # 评估指定模型
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    DEVICE, N_GROUPS, N_CROSS_SECTION_FEATURES, RANDOM_SEEDS,
)
from utils.dataset import split_dataset
from utils.metrics import calc_ic_series, ic_summary, group_return


def evaluate_model(
    model_name: str,
    checkpoint_path: str | None = None,
    seed: int = 42,
    window: int = 0,
) -> dict:
    """评估单个模型在测试集上的表现。

    Args:
        model_name: mlp/gbdt/gru/agru。
        checkpoint_path: 模型检查点路径（可选，自动查找）。
        seed: 随机种子。
        window: 滚动窗口索引（0-5），用于加载对应checkpoint和窗口数据。

    Returns:
        dict: 评估指标。
    """
    from config import ROLLING_WINDOWS

    # 加载数据
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"数据未找到: {processed_path}")

    df = pd.read_csv(processed_path, parse_dates=["date"])

    # 根据模型类型确定数据模式和标签列
    is_sequence = model_name in ("gru", "agru")
    if is_sequence:
        mode = "sequence"
        label_col = "label_vwap10"
        # 加载序列数据
        seq_path = PROCESSED_DATA_DIR / "sequences.npz"
        seq_dates_path = PROCESSED_DATA_DIR / "seq_dates.npz"
        if seq_path.exists() and seq_dates_path.exists():
            sequences_npz = np.load(seq_path, allow_pickle=True)
            sequences = {k: sequences_npz[k] for k in sequences_npz.files}
            seq_dates_npz = np.load(seq_dates_path, allow_pickle=True)
            seq_dates = {k: seq_dates_npz[k] for k in seq_dates_npz.files}
        else:
            raise FileNotFoundError(f"序列数据未找到: {seq_path}")
        seq_kwargs = {"sequences": sequences, "seq_dates": seq_dates}
    else:
        mode = "cross_section"
        label_col = "label"
        seq_kwargs = {}

    # 使用窗口中对应的切分参数
    if window < len(ROLLING_WINDOWS):
        train_end, val_end, test_end = ROLLING_WINDOWS[window]
    else:
        from config import TRAIN_END, VAL_END
        train_end, val_end = TRAIN_END, VAL_END
        test_end = None

    # 切分
    _, _, test_ds = split_dataset(
        df, train_end=train_end, val_end=val_end, test_end=test_end,
        mode=mode, label_col=label_col, **seq_kwargs,
    )

    # 加载模型
    if checkpoint_path:
        ckpt = checkpoint_path
    else:
        ckpt = str(CHECKPOINT_DIR / f"{model_name}_w{window}_s{seed}.pt")

    if model_name == "gbdt":
        import joblib
        ckpt = ckpt.replace(".pt", ".joblib")
        model = joblib.load(ckpt)
        is_gbdt = True
    else:
        if model_name == "mlp":
            from models.mlp_alpha import MLPAlphaModel
            model = MLPAlphaModel(n_factors=N_CROSS_SECTION_FEATURES)
        elif model_name == "gru":
            from models.gru_alpha import GRUAlphaModel
            model = GRUAlphaModel()
        elif model_name == "agru":
            from models.agru_alpha import AGRUAlphaModel
            model = AGRUAlphaModel()

        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        is_gbdt = False

    # 收集所有预测
    all_preds = []
    all_labels = []
    all_dates = []

    if is_gbdt:
        # GBDT: 一次性预测
        X_test_list, y_test_list = [], []
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            X_test_list.append(x.numpy())
            y_test_list.append(y.numpy())
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        preds = model.predict(X_test)
        all_preds = preds
        all_labels = y_test

        # 重建dates
        dates_list = []
        cnt = 0
        for i in range(len(test_ds)):
            x, _ = test_ds[i]
            n_stocks = len(x)
            date_val = test_ds.get_date(i)
            dates_list.extend([date_val] * n_stocks)
        all_dates = pd.Series(dates_list)

    else:
        with torch.no_grad():
            for i in range(len(test_ds)):
                x, y = test_ds[i]
                x = x.to(DEVICE)
                pred, _ = model(x)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.numpy())
                date_val = test_ds.get_date(i)
                all_dates.extend([date_val] * len(x))

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_dates = pd.Series(all_dates)

    # 计算IC序列
    pred_series = pd.Series(all_preds)
    label_series = pd.Series(all_labels)
    ic_series = calc_ic_series(pred_series, label_series, all_dates)
    ic_stats = ic_summary(ic_series)

    # 分组回测
    eval_df = pd.DataFrame({
        "date": all_dates,
        "prediction": all_preds,
        "label": all_labels,
    })
    groups = group_return(eval_df, n_groups=N_GROUPS)

    return {
        "model": model_name,
        "ic_stats": ic_stats,
        "ic_series": ic_series,
        "group_return": groups,
    }


def main():
    parser = argparse.ArgumentParser(description="Project-Beta Evaluation")
    parser.add_argument("--model", default="mlp", help="Model to evaluate")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--window", type=int, default=0, help="Rolling window index (0-5)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Evaluating {args.model} (W{args.window}, S{args.seed})...")
    result = evaluate_model(args.model, args.checkpoint, args.seed, window=args.window)

    print(f"\n{'='*50}")
    print(f"  Model: {result['model']}")
    print(f"  RankIC Mean:  {result['ic_stats']['rank_ic_mean']:.4f}")
    print(f"  ICIR:          {result['ic_stats']['icir']:.4f}")
    print(f"  IC Win Rate:   {result['ic_stats']['ic_win_rate']:.2%}")
    print(f"{'='*50}")

    print(f"\nGroup Return (Top 20 groups):")
    print(result["group_return"])

    # Save results
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results_path = LOG_DIR / f"eval_{args.model}_w{args.window}_s{args.seed}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": result["model"],
            "ic_stats": result["ic_stats"],
            "group_return": result["group_return"].to_dict(orient="records"),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
