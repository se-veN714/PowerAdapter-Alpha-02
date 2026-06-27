# -*- coding: utf-8 -*-
# @File    : run_ensemble.py
# @Time    : 2026/06/26
# @Project : Project-Beta (Alpha-02)

"""本模块提供了ICIR加权集成评估入口。

论文§2.3: 模型相关性分析与模型集成
- 对每个滚动窗口，生成4个模型在验证集上的预测
- 计算各模型滚动ICIR权重
- 构建ICIR加权集成因子
- 评估集成因子表现（RankIC/ICIR/胜率/分组回测）

子集成:
- CS Ensemble: MLP + GBDT（Close20标签）
- SEQ Ensemble: GRU + AGRU（VWAP10标签）
- Full Ensemble: 4模型（Close20标签评估）

使用方式:
    python run_ensemble.py
    python run_ensemble.py --no-log
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    DEVICE, N_CROSS_SECTION_FEATURES, N_GROUPS,
    RANDOM_SEEDS, ROLLING_WINDOWS, ICIR_WINDOW,
)
from ensemble import compute_rolling_icir, icir_weighted_voting
from utils.dataset import split_dataset
from utils.metrics import calc_ic_series, ic_summary, group_return, rank_ic


# ============================================================
# 预测生成
# ============================================================

def _load_sequence_data():
    """加载序列数据（GRU/AGRU用）。"""
    seq_path = PROCESSED_DATA_DIR / "sequences.npz"
    seq_dates_path = PROCESSED_DATA_DIR / "seq_dates.npz"
    if not seq_path.exists() or not seq_dates_path.exists():
        raise FileNotFoundError(f"序列数据未找到: {seq_path}")
    sequences_npz = np.load(seq_path, allow_pickle=True)
    sequences = {k: sequences_npz[k] for k in sequences_npz.files}
    seq_dates_npz = np.load(seq_dates_path, allow_pickle=True)
    seq_dates = {k: seq_dates_npz[k] for k in seq_dates_npz.files}
    return sequences, seq_dates


def generate_predictions(
    model_name: str,
    window: int,
    seed: int,
    df: pd.DataFrame,
    sequences: dict | None = None,
    seq_dates: dict | None = None,
    target_split: str = "val",
) -> pd.DataFrame:
    """为指定模型生成在val/test集上的预测。

    Args:
        model_name: mlp/gbdt/gru/agru
        window: 滚动窗口索引 (0-5)
        seed: 随机种子
        df: 预处理后的DataFrame
        sequences: 序列数据（GRU/AGRU用）
        seq_dates: 序列日期（GRU/AGRU用）
        target_split: "val" 或 "test"

    Returns:
        DataFrame [date, stock_code, prediction, label]
    """
    train_end, val_end, test_end = ROLLING_WINDOWS[window]

    is_sequence = model_name in ("gru", "agru")

    if is_sequence:
        mode = "sequence"
        label_col = "label_vwap10"
        seq_kwargs = {"sequences": sequences, "seq_dates": seq_dates}
    else:
        mode = "cross_section"
        label_col = "label"
        seq_kwargs = {}

    # 数据切分
    train_ds, val_ds, test_ds = split_dataset(
        df, train_end=train_end, val_end=val_end, test_end=test_end,
        mode=mode, label_col=label_col, **seq_kwargs,
    )

    target_ds = val_ds if target_split == "val" else test_ds

    # 加载模型
    if model_name == "gbdt":
        import joblib
        ckpt = str(CHECKPOINT_DIR / f"{model_name}_w{window}_s{seed}.joblib")
        model = joblib.load(ckpt)
        is_gbdt = True
    else:
        ckpt = str(CHECKPOINT_DIR / f"{model_name}_w{window}_s{seed}.pt")
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

    # 生成预测
    records = []

    if is_gbdt:
        # GBDT: 一次性预测
        X_list, y_list = [], []
        dates_list = []
        for i in range(len(target_ds)):
            x, y = target_ds[i]
            X_list.append(x.numpy())
            y_list.append(y.numpy())
            date_val = target_ds.get_date(i)
            dates_list.append(date_val)

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        preds = model.predict(X_all)

        # 获取stock_codes
        stock_codes = []
        for i in range(len(target_ds)):
            date_val = target_ds.get_date(i)
            section = df[df["date"] == date_val]
            if len(section) > 0:
                codes = section["stock_code"].values
                stock_codes.extend(codes)

        # 确保长度匹配
        min_len = min(len(preds), len(stock_codes))
        preds = preds[:min_len]
        stock_codes = stock_codes[:min_len]
        y_all = y_all[:min_len]

        # 重建dates
        expanded_dates = []
        cnt = 0
        for i in range(len(target_ds)):
            date_val = target_ds.get_date(i)
            n = len(target_ds[i][0])
            expanded_dates.extend([date_val] * n)
        expanded_dates = expanded_dates[:min_len]

        for j in range(min_len):
            records.append({
                "date": pd.Timestamp(expanded_dates[j]),
                "stock_code": str(stock_codes[j]),
                "prediction": float(preds[j]),
                "label": float(y_all[j]),
            })
    else:
        with torch.no_grad():
            for i in range(len(target_ds)):
                x, y = target_ds[i]
                x = x.to(DEVICE)
                pred, _ = model(x)
                pred_np = pred.cpu().numpy()
                label_np = y.numpy()
                date_val = target_ds.get_date(i)

                # 获取stock_codes
                section = df[df["date"] == pd.Timestamp(date_val)]
                codes = section["stock_code"].values

                for j in range(min(len(pred_np), len(codes), len(label_np))):
                    records.append({
                        "date": pd.Timestamp(date_val),
                        "stock_code": str(codes[j]),
                        "prediction": float(pred_np[j]),
                        "label": float(label_np[j]),
                    })

    return pd.DataFrame(records)


# ============================================================
# ICIR加权集成
# ============================================================

def compute_model_ic_history(
    model_pred_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """为每个模型计算每日RankIC序列（用于ICIR权重）。

    每个模型使用自己的预测和标签计算IC。

    Args:
        model_pred_dfs: {model_name: DataFrame[date, stock_code, prediction, label]}

    Returns:
        {model_name: IC Series (日期索引)}
    """
    ic_histories = {}
    for name, df in model_pred_dfs.items():
        ic_series = calc_ic_series(df["prediction"], df["label"], df["date"])
        ic_histories[name] = ic_series
    return ic_histories


def build_ensemble_daily(
    model_pred_dfs: dict[str, pd.DataFrame],
    ic_histories: dict[str, pd.Series],
    window: int = ICIR_WINDOW,
) -> pd.DataFrame:
    """构建逐日ICIR加权集成预测。

    Args:
        model_pred_dfs: {model_name: DataFrame[date, stock_code, prediction, label]}
        ic_histories: {model_name: IC Series}
        window: ICIR窗口大小

    Returns:
        DataFrame [date, stock_code, ensemble_pred, label]（label取自首模型）
    """
    all_dates = set()
    for df in model_pred_dfs.values():
        all_dates.update(df["date"].unique())
    all_dates = sorted(all_dates)

    first_model = list(model_pred_dfs.keys())[0]
    ensemble_records = []

    for date in all_dates:
        # 计算当前日期的ICIR权重（用截至前一日的IC）
        model_icirs = {}
        for name, ic_series in ic_histories.items():
            past_ic = ic_series[ic_series.index < date].tail(window)
            if len(past_ic) < window // 2:
                model_icirs[name] = 0.01
            else:
                mean_ic = past_ic.mean()
                std_ic = past_ic.std()
                model_icirs[name] = mean_ic / std_ic if std_ic > 0 else 0.01

        # 获取当日各模型预测
        date_preds = {}
        for name, df in model_pred_dfs.items():
            day_data = df[df["date"] == date]
            if len(day_data) > 0:
                date_preds[name] = day_data

        if len(date_preds) < 2:
            continue

        # 对齐股票
        common_stocks = None
        for name, day_data in date_preds.items():
            stocks = set(day_data["stock_code"].values)
            common_stocks = stocks if common_stocks is None else common_stocks & stocks

        if not common_stocks or len(common_stocks) < 3:
            continue

        # ICIR加权集成
        aligned_preds = {}
        for name, day_data in date_preds.items():
            day_data_idx = day_data.set_index("stock_code")
            preds = day_data_idx.loc[list(common_stocks)]["prediction"].values
            aligned_preds[name] = preds

        ensemble_pred = icir_weighted_voting(aligned_preds, model_icirs)

        # label取自首模型
        first_day = model_pred_dfs[first_model][model_pred_dfs[first_model]["date"] == date]
        label_map = dict(zip(first_day["stock_code"], first_day["label"]))

        for stock, pred in zip(sorted(common_stocks), ensemble_pred):
            ensemble_records.append({
                "date": date,
                "stock_code": stock,
                "prediction": pred,
                "label": label_map.get(stock, float("nan")),
            })

    return pd.DataFrame(ensemble_records)


# ============================================================
# 主评估流程
# ============================================================

def evaluate_ensemble_window(
    window: int,
    seed: int,
    df: pd.DataFrame,
    sequences: dict,
    seq_dates: dict,
    print_fn=print,
) -> dict:
    """评估单个窗口×种子的集成表现。

    Returns:
        dict: 包含各个集成和子集成的评估指标
    """
    t0 = time.time()
    train_end, val_end, test_end = ROLLING_WINDOWS[window]
    print_fn(f"\n{'='*50}")
    print_fn(f"  Window {window}: train_end={train_end}, val={val_end}, test={test_end}")
    print_fn(f"  Seed: {seed}")
    print_fn(f"{'='*50}")

    # --- Step 1: 生成各模型在验证集上的预测 ---
    print_fn(f"\n[1/4] Generating val predictions...")

    val_preds = {}
    for model_name in ["mlp", "gbdt", "gru", "agru"]:
        t_m = time.time()
        pred_df = generate_predictions(
            model_name, window, seed, df,
            sequences=sequences, seq_dates=seq_dates,
            target_split="val",
        )
        val_preds[model_name] = pred_df
        print_fn(f"  {model_name.upper():5s}: {len(pred_df):6d} preds, {pred_df['date'].nunique():4d} dates ({time.time()-t_m:.1f}s)")

    # --- Step 2: 计算验证集IC历史 ---
    print_fn(f"\n[2/4] Computing val IC histories...")
    ic_histories = compute_model_ic_history(val_preds)
    for name, ic_series in ic_histories.items():
        print_fn(f"  {name.upper():5s} Val IC: mean={ic_series.mean():.4f}, "
                 f"std={ic_series.std():.4f}, win_rate={(ic_series>0).mean():.2%}")

    # --- Step 3: 生成测试集预测 + 构建集成 ---
    print_fn(f"\n[3/4] Generating test predictions & building ensembles...")

    test_preds = {}
    for model_name in ["mlp", "gbdt", "gru", "agru"]:
        t_m = time.time()
        pred_df = generate_predictions(
            model_name, window, seed, df,
            sequences=sequences, seq_dates=seq_dates,
            target_split="test",
        )
        test_preds[model_name] = pred_df
        print_fn(f"  {model_name.upper():5s}: {len(pred_df):6d} preds, {pred_df['date'].nunique():4d} dates ({time.time()-t_m:.1f}s)")

    # 构建子集成
    # CS Ensemble: MLP + GBDT
    print_fn(f"\n  Building CS Ensemble (MLP+GBDT)...")
    cs_test_preds = {"mlp": test_preds["mlp"], "gbdt": test_preds["gbdt"]}
    cs_ic_hist = {"mlp": ic_histories["mlp"], "gbdt": ic_histories["gbdt"]}
    cs_ensemble_df = build_ensemble_daily(cs_test_preds, cs_ic_hist)

    # SEQ Ensemble: GRU + AGRU
    print_fn(f"  Building SEQ Ensemble (GRU+AGRU)...")
    seq_test_preds = {"gru": test_preds["gru"], "agru": test_preds["agru"]}
    seq_ic_hist = {"gru": ic_histories["gru"], "agru": ic_histories["agru"]}
    seq_ensemble_df = build_ensemble_daily(seq_test_preds, seq_ic_hist)

    # Full Ensemble: 4 models
    print_fn(f"  Building Full Ensemble (4 models)...")
    full_ensemble_df = build_ensemble_daily(test_preds, ic_histories)

    # --- Step 4: 评估 ---
    print_fn(f"\n[4/4] Evaluating ensembles...")
    results = {}

    # 单模型测试集IC
    single_results = {}
    for name in ["mlp", "gbdt", "gru", "agru"]:
        ic_s = calc_ic_series(test_preds[name]["prediction"], test_preds[name]["label"], test_preds[name]["date"])
        stats = ic_summary(ic_s)
        single_results[name] = stats
        print_fn(f"  {name.upper():5s} Test: RankIC={stats['rank_ic_mean']:.4f}, "
                 f"ICIR={stats['icir']:.4f}, WinRate={stats['ic_win_rate']:.2%}")

    # CS Ensemble
    if len(cs_ensemble_df) > 0:
        cs_ic = calc_ic_series(cs_ensemble_df["prediction"], cs_ensemble_df["label"], cs_ensemble_df["date"])
        cs_stats = ic_summary(cs_ic)
        cs_groups = group_return(cs_ensemble_df, n_groups=N_GROUPS)
        results["cs_ensemble"] = {
            "ic_stats": cs_stats,
            "group_return": cs_groups.to_dict(orient="records"),
            "n_dates": int(cs_ensemble_df["date"].nunique()),
        }
        print_fn(f"  CS-E   Test: RankIC={cs_stats['rank_ic_mean']:.4f}, "
                 f"ICIR={cs_stats['icir']:.4f}, WinRate={cs_stats['ic_win_rate']:.2%}")

    # SEQ Ensemble
    if len(seq_ensemble_df) > 0:
        seq_ic = calc_ic_series(seq_ensemble_df["prediction"], seq_ensemble_df["label"], seq_ensemble_df["date"])
        seq_stats = ic_summary(seq_ic)
        seq_groups = group_return(seq_ensemble_df, n_groups=N_GROUPS)
        results["seq_ensemble"] = {
            "ic_stats": seq_stats,
            "group_return": seq_groups.to_dict(orient="records"),
            "n_dates": int(seq_ensemble_df["date"].nunique()),
        }
        print_fn(f"  SEQ-E  Test: RankIC={seq_stats['rank_ic_mean']:.4f}, "
                 f"ICIR={seq_stats['icir']:.4f}, WinRate={seq_stats['ic_win_rate']:.2%}")

    # Full Ensemble
    if len(full_ensemble_df) > 0:
        full_ic = calc_ic_series(full_ensemble_df["prediction"], full_ensemble_df["label"], full_ensemble_df["date"])
        full_stats = ic_summary(full_ic)
        full_groups = group_return(full_ensemble_df, n_groups=N_GROUPS)
        results["full_ensemble"] = {
            "ic_stats": full_stats,
            "group_return": full_groups.to_dict(orient="records"),
            "n_dates": int(full_ensemble_df["date"].nunique()),
        }
        print_fn(f"  FULL-E Test: RankIC={full_stats['rank_ic_mean']:.4f}, "
                 f"ICIR={full_stats['icir']:.4f}, WinRate={full_stats['ic_win_rate']:.2%}")

    results["single_models"] = single_results
    results["ic_histories"] = {k: v.to_dict() for k, v in ic_histories.items()}
    results["elapsed_s"] = round(time.time() - t0, 1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Project-Beta ICIR Ensemble Evaluation")
    parser.add_argument("--no-log", action="store_true", help="Disable dual-output logging")
    parser.add_argument("--windows", type=str, default="0,1,2,3,4,5",
                       help="Comma-separated window indices (default: all 6)")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                       help="Comma-separated seeds (default: all 3)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    windows = [int(w.strip()) for w in args.windows.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # ===== 日志双路输出 =====
    log_path = None
    if not args.no_log:
        log_dir = r"D:\.shigodo\shigodo\Quantification\wb-remote-monitor\logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"ensemble_{ts}.log"

    def run():
        return _run_ensemble(windows, seeds)

    if log_path:
        sys.path.insert(0, r"D:\.shigodo\shigodo\Quantification\wb-remote-monitor")
        from log_to_file import log_writer
        with log_writer(str(log_path)):
            run()
        print(f"\n[INFO] Log saved to: {log_path}")
    else:
        run()


def _run_ensemble(windows: list[int], seeds: list[int]):
    """执行ICIR加权集成评估。"""
    print("=" * 60)
    print("  Project-Beta ICIR Weighted Ensemble Evaluation")
    print(f"  Windows: {windows}")
    print(f"  Seeds:   {seeds}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 加载数据
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print(f"[ERROR] Data not found: {processed_path}")
        return

    df = pd.read_csv(processed_path, parse_dates=["date"])
    print(f"\nData loaded: {len(df)} rows, {df['date'].nunique()} dates")

    # 加载序列数据
    sequences, seq_dates = _load_sequence_data()
    print(f"Sequences loaded: {len(sequences)} stocks")

    # 遍历所有窗口×种子
    total_tasks = len(windows) * len(seeds)
    print(f"\nTotal tasks: {total_tasks} (W{len(windows)} × S{len(seeds)})")

    all_results = {}  # {f"w{w}_s{s}": result}

    task_idx = 0
    for w in windows:
        for s in seeds:
            task_idx += 1
            print(f"\n{'#'*50}")
            print(f"  Task {task_idx}/{total_tasks}: Window={w}, Seed={s}")
            print(f"{'#'*50}")

            result = evaluate_ensemble_window(w, s, df, sequences, seq_dates)
            all_results[f"w{w}_s{s}"] = result

    # ===== 汇总 =====
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY: ICIR Ensemble Evaluation")
    print(f"{'='*60}")

    # 按集成类型汇总
    ensemble_types = ["cs_ensemble", "seq_ensemble", "full_ensemble"]
    model_names = ["mlp", "gbdt", "gru", "agru"]

    summary = {}

    # 单模型汇总
    for model_name in model_names:
        ics = []
        for key, r in all_results.items():
            if "single_models" in r and model_name in r["single_models"]:
                ics.append(r["single_models"][model_name]["rank_ic_mean"])
        if ics:
            summary[f"single_{model_name}"] = {
                "mean_ic": round(np.mean(ics), 6),
                "std_ic": round(np.std(ics), 6),
                "n_tasks": len(ics),
            }
            print(f"\n  {model_name.upper():5s}: Mean IC={np.mean(ics):.4f} ± {np.std(ics):.4f} ({len(ics)} tasks)")

    # 集成汇总
    for etype in ensemble_types:
        ics = []
        icirs = []
        win_rates = []
        n_dates_list = []
        for key, r in all_results.items():
            if etype in r:
                ics.append(r[etype]["ic_stats"]["rank_ic_mean"])
                icirs.append(r[etype]["ic_stats"]["icir"])
                win_rates.append(r[etype]["ic_stats"]["ic_win_rate"])
                n_dates_list.append(r[etype].get("n_dates", 0))

        if ics:
            summary[etype] = {
                "mean_ic": round(np.mean(ics), 6),
                "std_ic": round(np.std(ics), 6),
                "mean_icir": round(np.mean(icirs), 6),
                "mean_win_rate": round(np.mean(win_rates), 6),
                "mean_n_dates": round(np.mean(n_dates_list), 1),
                "n_tasks": len(ics),
            }
            label_map = {"cs_ensemble": "CS-E   (MLP+GBDT)", "seq_ensemble": "SEQ-E  (GRU+AGRU)", "full_ensemble": "FULL-E (4 models)"}
            label = label_map.get(etype, etype)
            print(f"\n  {label}:")
            print(f"    Mean IC = {np.mean(ics):.4f} ± {np.std(ics):.4f}")
            print(f"    ICIR    = {np.mean(icirs):.4f}")
            print(f"    Win Rate= {np.mean(win_rates):.2%}")
            print(f"    N Dates = {np.mean(n_dates_list):.0f}")

    # 保存结果
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = LOG_DIR / f"ensemble_results_{timestamp}.json"

    # 清理 per_task 中的 ic_histories（Timestamp keys 不能直接序列化）
    clean_tasks = {}
    for key, task in all_results.items():
        clean_task = {}
        for k, v in task.items():
            if k == "ic_histories":
                clean_task[k] = {m: {str(d): ic for d, ic in series.items()}
                                for m, series in v.items()}
            else:
                clean_task[k] = v
        clean_tasks[key] = clean_task

    save_data = {
        "config": {
            "windows": windows,
            "seeds": seeds,
            "icir_window": ICIR_WINDOW,
            "timestamp": timestamp,
        },
        "per_task": clean_tasks,
        "summary": summary,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n\nResults saved to: {results_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
