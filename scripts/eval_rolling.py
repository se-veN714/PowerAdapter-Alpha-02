# -*- coding: utf-8 -*-
# @File    : eval_rolling.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""滚动训练后的评估入口。

功能：
1. 加载 4 模型 × 6 窗口 × 3 种子的滚动训练检查点
2. 对每个窗口选择验证 IC 最优的种子，生成该窗口测试集预测
3. 拼接 6 个窗口测试预测，得到每个模型连续的测试集预测
4. 计算每个模型逐日 RankIC 序列，并构造 60 日滚动 ICIR
5. 按 ICIR 加权 Voting 构建集成因子
6. 输出单模型 + 集成因子的 RankIC/ICIR/IC胜率/20 组分组回测

使用方式：
    python scripts/eval_rolling.py

输出：
    logs/eval_rolling_results.json
    logs/eval_rolling_groups.csv
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    DEVICE, N_CROSS_SECTION_FEATURES,
    ROLLING_WINDOWS, RANDOM_SEEDS, BUFFER_DAYS, ICIR_WINDOW, N_GROUPS,
)
from ensemble import build_ensemble_predictions
from utils.dataset import split_dataset
from utils.metrics import calc_ic_series, ic_summary, group_return, rank_ic


def section_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对序列输入按截面做 zscore 标准化。"""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    return (x - mean) / (std + eps)


def load_model(model_name: str, window_idx: int, seed: int):
    """加载指定模型、窗口、种子的检查点。"""
    if model_name == "gbdt":
        ckpt_path = CHECKPOINT_DIR / f"gbdt_w{window_idx}_s{seed}.joblib"
        if not ckpt_path.exists():
            return None
        return joblib.load(ckpt_path)

    if model_name == "mlp":
        from models.mlp_alpha import MLPAlphaModel
        model = MLPAlphaModel(n_factors=N_CROSS_SECTION_FEATURES)
    elif model_name == "gru":
        from models.gru_alpha import GRUAlphaModel
        model = GRUAlphaModel()
    elif model_name == "agru":
        from models.agru_alpha import AGRUAlphaModel
        model = AGRUAlphaModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt_path = CHECKPOINT_DIR / f"{model_name}_w{window_idx}_s{seed}.pt"
    if not ckpt_path.exists():
        return None

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_seed_on_val(model, val_ds, model_name: str) -> float:
    """在验证集上计算 RankIC，用于选择最优种子。"""
    all_preds = []
    all_labels = []

    if model_name == "gbdt":
        X_val_list, y_val_list = [], []
        for i in range(len(val_ds)):
            x, y = val_ds[i]
            X_val_list.append(x.numpy())
            y_val_list.append(y.numpy())
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        all_preds = model.predict(X_val)
        all_labels = y_val
    else:
        with torch.no_grad():
            for i in range(len(val_ds)):
                x, y = val_ds[i]
                x = x.to(DEVICE)
                if x.dim() in (3, 4) and x.shape[0] == 1:
                    x = x.squeeze(0)
                if y.dim() == 2 and y.shape[0] == 1:
                    y = y.squeeze(0)
                if x.shape[0] == 0:
                    continue
                if x.dim() == 3:
                    x = section_zscore(x)
                pred, _ = model(x)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

    return rank_ic(all_preds, all_labels)


def generate_test_predictions(model, test_ds, model_name: str) -> pd.DataFrame:
    """生成测试集预测 DataFrame [date, stock_code, prediction, label]。"""
    records = []

    if model_name == "gbdt":
        X_test_list, y_test_list = [], []
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            X_test_list.append(x.numpy())
            y_test_list.append(y.numpy())
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        preds = model.predict(X_test)

        cnt = 0
        for i in range(len(test_ds)):
            x, _ = test_ds[i]
            n = len(x)
            date_val = test_ds.get_date(i)
            section = test_ds.data[test_ds.data["date"] == date_val]
            codes = section["stock_code"].values
            for j in range(n):
                if cnt + j >= len(preds):
                    break
                records.append({
                    "date": date_val,
                    "stock_code": str(codes[j]),
                    "prediction": float(preds[cnt + j]),
                    "label": float(y_test[cnt + j]),
                })
            cnt += n
    else:
        with torch.no_grad():
            for i in range(len(test_ds)):
                x, y = test_ds[i]
                x = x.to(DEVICE)
                if x.dim() in (3, 4) and x.shape[0] == 1:
                    x = x.squeeze(0)
                if y.dim() == 2 and y.shape[0] == 1:
                    y = y.squeeze(0)
                if x.shape[0] == 0:
                    continue
                if x.dim() == 3:
                    x = section_zscore(x)
                pred, _ = model(x)
                pred = pred.cpu().numpy()

                date_val = test_ds.get_date(i)
                section = test_ds.data[test_ds.data["date"] == date_val]
                codes = section["stock_code"].values
                for j in range(len(pred)):
                    records.append({
                        "date": date_val,
                        "stock_code": str(codes[j]),
                        "prediction": float(pred[j]),
                        "label": float(y[j].item() if torch.is_tensor(y[j]) else y[j]),
                    })

    return pd.DataFrame(records)


def select_best_seed(model_name: str, window_idx: int, val_ds) -> tuple[int, float]:
    """选择验证 IC 最高的种子。"""
    best_seed = RANDOM_SEEDS[0]
    best_val_ic = -float("inf")

    for seed in RANDOM_SEEDS:
        model = load_model(model_name, window_idx, seed)
        if model is None:
            continue
        val_ic = evaluate_seed_on_val(model, val_ds, model_name)
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_seed = seed

    return best_seed, best_val_ic


def collect_model_predictions(
    model_name: str,
    df: pd.DataFrame,
    sequences: dict[str, np.ndarray] | None,
    seq_dates: dict[str, np.ndarray] | None,
) -> pd.DataFrame:
    """收集某模型在 6 个窗口测试集上的预测。"""
    all_records = []
    mode = "sequence" if model_name in ("gru", "agru") else "cross_section"

    for w, (train_end, val_end, test_end) in enumerate(ROLLING_WINDOWS):
        print(f"  Window {w}: {train_end} / {val_end} / {test_end}")

        if mode == "cross_section":
            train_ds, val_ds, test_ds = split_dataset(
                df, train_end=train_end, val_end=val_end, test_end=test_end, mode="cross_section"
            )
        else:
            train_ds, val_ds, test_ds = split_dataset(
                df, train_end=train_end, val_end=val_end, test_end=test_end,
                mode="sequence", sequences=sequences, seq_dates=seq_dates,
            )

        if len(test_ds) == 0:
            print(f"    [WARN] 测试集为空，跳过")
            continue

        best_seed, best_val_ic = select_best_seed(model_name, w, val_ds)
        print(f"    Best seed={best_seed}, val_ic={best_val_ic:.4f}")

        model = load_model(model_name, w, best_seed)
        if model is None:
            print(f"    [WARN] 检查点不存在，跳过")
            continue

        pred_df = generate_test_predictions(model, test_ds, model_name)
        pred_df["model"] = model_name
        pred_df["window"] = w
        all_records.append(pred_df)

    return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("Rolling Evaluation: 4 models × 6 windows")
    print("=" * 60)

    # 加载数据
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    df = pd.read_csv(processed_path, parse_dates=["date"])
    print(f"Data loaded: {len(df)} rows, {df['date'].nunique()} dates")

    # 加载序列数据
    seq_path = PROCESSED_DATA_DIR / "sequences.npz"
    seq_dates_path = PROCESSED_DATA_DIR / "seq_dates.npz"
    sequences, seq_dates = None, None
    if seq_path.exists() and seq_dates_path.exists():
        seq_npz = np.load(seq_path, allow_pickle=True)
        sequences = {k: seq_npz[k] for k in seq_npz.files}
        seq_dates_npz = np.load(seq_dates_path, allow_pickle=True)
        seq_dates = {k: seq_dates_npz[k] for k in seq_dates_npz.files}
        print("Sequence data loaded")
    else:
        print("[WARN] Sequence data not found, GRU/AGRU will be skipped")

    model_names = ["mlp", "gbdt", "gru", "agru"]
    model_pred_dfs = {}
    ic_history = {}

    for model_name in model_names:
        print(f"\n{'='*40}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*40}")

        if model_name in ("gru", "agru") and sequences is None:
            print(f"  [SKIP] 无序列数据")
            continue

        pred_df = collect_model_predictions(model_name, df, sequences, seq_dates)
        if pred_df.empty:
            print(f"  [WARN] 无预测结果")
            continue

        model_pred_dfs[model_name] = pred_df[["date", "stock_code", "prediction", "label"]].copy()

        # 计算逐日 IC 序列
        ic_series = calc_ic_series(pred_df["prediction"], pred_df["label"], pred_df["date"])
        ic_history[model_name] = ic_series

        summary = ic_summary(ic_series)
        print(f"  Test IC: rank_ic_mean={summary['rank_ic_mean']:.4f}, "
              f"icir={summary['icir']:.4f}, win_rate={summary['ic_win_rate']:.2%}")

    # ICIR 加权集成
    print(f"\n{'='*40}")
    print("ICIR Weighted Ensemble")
    print(f"{'='*40}")

    ensemble_df = build_ensemble_predictions(model_pred_dfs, ic_history, window=ICIR_WINDOW)
    if ensemble_df.empty:
        print("[WARN] 集成预测为空")
    else:
        ensemble_ic = calc_ic_series(ensemble_df["prediction"], ensemble_df["label"], ensemble_df["date"])
        ensemble_summary = ic_summary(ensemble_ic)
        print(f"  Ensemble IC: rank_ic_mean={ensemble_summary['rank_ic_mean']:.4f}, "
              f"icir={ensemble_summary['icir']:.4f}, win_rate={ensemble_summary['ic_win_rate']:.2%}")

    # 分组回测
    print(f"\n{'='*40}")
    print("Group Backtest (20 groups)")
    print(f"{'='*40}")

    results = {}
    for name, pred_df in model_pred_dfs.items():
        groups = group_return(pred_df, n_groups=N_GROUPS)
        print(f"\n{name.upper()}:")
        print(groups.to_string(index=False))
        results[name] = {
            "ic_summary": ic_summary(calc_ic_series(pred_df["prediction"], pred_df["label"], pred_df["date"])),
            "group_return": groups.to_dict(orient="records"),
        }

    if not ensemble_df.empty:
        ensemble_groups = group_return(ensemble_df, n_groups=N_GROUPS)
        print(f"\nENSEMBLE:")
        print(ensemble_groups.to_string(index=False))
        results["ensemble"] = {
            "ic_summary": ensemble_summary,
            "group_return": ensemble_groups.to_dict(orient="records"),
        }

    # 保存结果
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results_path = LOG_DIR / "eval_rolling_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    if not ensemble_df.empty:
        groups_path = LOG_DIR / "eval_rolling_groups.csv"
        ensemble_groups.to_csv(groups_path, index=False, encoding="utf-8-sig")
        print(f"Ensemble group return saved to {groups_path}")


if __name__ == "__main__":
    main()
