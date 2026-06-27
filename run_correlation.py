# -*- coding: utf-8 -*-
# @File    : run_correlation.py
# @Time    : 2026/06/26
# @Project : Project-Beta (Alpha-02)

"""模型相关性分析 — 论文表8: 截面相关系数矩阵。

遍历所有滚动窗口×种子，生成4模型在测试集上的预测，
计算逐日截面Rank相关系数，汇总为平均相关系数矩阵。

使用方式:
    python run_correlation.py
    python run_correlation.py --no-log
    python run_correlation.py --no-log --split test  # 测试集（默认）
    python run_correlation.py --no-log --split val   # 验证集
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    N_CROSS_SECTION_FEATURES, RANDOM_SEEDS, ROLLING_WINDOWS,
)
from run_ensemble import generate_predictions, _load_sequence_data


# ============================================================
# 核心：截面相关系数
# ============================================================

def compute_daily_cross_section_corr(
    pred_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """对每个模型对的每日截面预测计算Rank相关系数。

    Args:
        pred_dfs: {model_name: DataFrame[date, stock_code, prediction]}

    Returns:
        DataFrame: 日期×模型对（列名为 "MLP_vs_GBDT" 等），值为相关系数。
    """
    # 获取共同日期
    common_dates = None
    for df in pred_dfs.values():
        dates = set(df["date"].unique())
        common_dates = dates if common_dates is None else common_dates & dates

    common_dates = sorted(common_dates)
    if not common_dates:
        print("[WARN] 无共同日期，跳过相关性计算")
        return pd.DataFrame()

    models = sorted(pred_dfs.keys())
    pairs = [(models[i], models[j]) for i in range(len(models))
             for j in range(i + 1, len(models))]

    records = []
    for date in common_dates:
        # 提取当日各模型预测（对齐股票）
        date_preds = {}
        for name, df in pred_dfs.items():
            day = df[df["date"] == date].set_index("stock_code")["prediction"]
            date_preds[name] = day

        # 找到共同股票
        common_stocks = None
        for name in models:
            stocks = set(date_preds[name].index)
            common_stocks = stocks if common_stocks is None else common_stocks & stocks

        if not common_stocks or len(common_stocks) < 5:
            continue

        row = {"date": date}
        for a, b in pairs:
            pa = date_preds[a].loc[list(common_stocks)].values
            pb = date_preds[b].loc[list(common_stocks)].values
            # 处理常数数组（Spearman无法计算）
            if np.std(pa) < 1e-12 or np.std(pb) < 1e-12:
                row[f"{a}_vs_{b}"] = 0.0
            else:
                corr, _ = spearmanr(pa, pb)
                row[f"{a}_vs_{b}"] = corr if not np.isnan(corr) else 0.0
        records.append(row)

    return pd.DataFrame(records)


def corr_matrix_from_pairs(corr_df: pd.DataFrame, models: list[str]) -> tuple:
    """从逐日pairwise相关系数构建平均相关系数矩阵。

    动态检测列名（兼容两种命名顺序），支持缺失模型。

    Args:
        corr_df: 日期×模型对 DataFrame
        models: 模型名列表（用于确定矩阵维度）

    Returns:
        (mean_matrix, std_matrix) N×N DataFrame
    """
    # 从 DataFrame 列动态解析模型对映射
    pair_map: dict[tuple[str, str], str] = {}
    for col in corr_df.columns:
        if col == "date":
            continue
        if "_vs_" in col:
            a, b = col.split("_vs_")
            pair_map[(a, b)] = col
            pair_map[(b, a)] = col  # 双向映射

    n = len(models)
    mean_matrix = pd.DataFrame(np.eye(n), index=models, columns=models)
    std_matrix = pd.DataFrame(np.zeros((n, n)), index=models, columns=models)

    for i in range(n):
        for j in range(i + 1, n):
            # 尝试两种顺序
            col = pair_map.get((models[i], models[j]))
            if col and col in corr_df.columns:
                vals = corr_df[col].dropna()
                mean_val = float(np.nan_to_num(vals.mean(), nan=0.0))
                std_val = float(np.nan_to_num(vals.std(), nan=0.0))
                mean_matrix.loc[models[i], models[j]] = mean_val
                mean_matrix.loc[models[j], models[i]] = mean_val
                std_matrix.loc[models[i], models[j]] = std_val
                std_matrix.loc[models[j], models[i]] = std_val

    return mean_matrix, std_matrix


# ============================================================
# 主流程
# ============================================================

MODELS = ["mlp", "gbdt", "gru", "agru"]


def run_all(
    df: pd.DataFrame,
    sequences: dict,
    seq_dates: dict,
    windows: list[int],
    seeds: list[int],
    target_split: str = "test",
) -> dict:
    """遍历所有窗口×种子，计算4模型相关系数矩阵。

    Returns:
        {
            "mean_matrix": DataFrame,      # 全部分汇总平均矩阵
            "std_matrix": DataFrame,       # 全部分汇总标准差矩阵
            "per_window": {                # 按窗口汇总
                "w0": {"mean": DataFrame, "std": DataFrame, "n_tasks": int, "n_dates": int},
                ...
            },
            "per_seed": {                  # 按种子汇总
                "42": {"mean": DataFrame, ...},
                ...
            },
            "per_task": {                  # 每个(W,S)任务
                "w0_s42": {"mean_matrix": ..., "n_dates": ...},
                ...
            },
        }
    """
    n_total = len(windows) * len(seeds)
    all_pair_dfs = {}   # key: wS_sS, value: pair DataFrame
    n_done = 0

    for w in windows:
        for s in seeds:
            key = f"w{w}_s{s}"
            print(f"\n{'='*60}")
            print(f"[{n_done+1}/{n_total}] Window={w}, Seed={s}")
            print(f"{'='*60}")

            # 生成4模型预测
            pred_dfs = {}
            for model in MODELS:
                print(f"  Generating {model.upper()} predictions...", end=" ")
                try:
                    pred_df = generate_predictions(
                        model, w, s, df, sequences, seq_dates,
                        target_split=target_split,
                    )
                    pred_dfs[model] = pred_df[["date", "stock_code", "prediction"]]
                    print(f"{len(pred_df)} rows, {pred_df['date'].nunique()} dates")
                except FileNotFoundError:
                    print(f"SKIP (checkpoint not found)")
                except Exception as e:
                    print(f"ERROR: {e}")

            if len(pred_dfs) < 2:
                print(f"  [SKIP] 可用模型不足: {list(pred_dfs.keys())}")
                n_done += 1
                continue

            # 计算逐日pairwise相关系数
            print(f"  Computing pairwise correlations...")
            pair_df = compute_daily_cross_section_corr(pred_dfs)
            all_pair_dfs[key] = pair_df
            n_done += 1

            if len(pair_df) > 0:
                models_used = sorted(pred_dfs.keys())
                mean_mat, std_mat = corr_matrix_from_pairs(pair_df, models_used)
                print(f"  → {len(pair_df)} dates, mean corr matrix:")
                print(mean_mat.to_string(float_format=lambda x: f"{x:.4f}"))
            else:
                print(f"  → 无有效日期")

    print(f"\n\n{'='*60}")
    print(f"汇总: {len(all_pair_dfs)} 个任务完成")

    # 汇总
    # --- 全部汇总 ---
    all_pairs = pd.concat(all_pair_dfs.values(), ignore_index=True)
    global_mean, global_std = corr_matrix_from_pairs(all_pairs, MODELS)

    # --- 按窗口汇总 ---
    per_window = {}
    for w in windows:
        window_pairs = []
        for key, pdf in all_pair_dfs.items():
            if key.startswith(f"w{w}_"):
                window_pairs.append(pdf)
        if window_pairs:
            w_df = pd.concat(window_pairs, ignore_index=True)
            w_mean, w_std = corr_matrix_from_pairs(w_df, MODELS)
            per_window[f"w{w}"] = {
                "mean": w_mean.to_dict(),
                "std": w_std.to_dict(),
                "n_tasks": len(window_pairs),
                "n_dates": len(w_df),
            }

    # --- 按种子汇总 ---
    per_seed = {}
    for s in seeds:
        seed_pairs = []
        for key, pdf in all_pair_dfs.items():
            if key.endswith(f"_s{s}"):
                seed_pairs.append(pdf)
        if seed_pairs:
            s_df = pd.concat(seed_pairs, ignore_index=True)
            s_mean, s_std = corr_matrix_from_pairs(s_df, MODELS)
            per_seed[f"s{s}"] = {
                "mean": s_mean.to_dict(),
                "std": s_std.to_dict(),
                "n_tasks": len(seed_pairs),
                "n_dates": len(s_df),
            }

    # --- 每个任务 ---
    per_task = {}
    for key, pdf in all_pair_dfs.items():
        models_used_in_task = set()
        for col in pdf.columns:
            if col == "date":
                continue
            models_used_in_task.update(col.split("_vs_"))
        m_list = sorted(models_used_in_task)
        m_mean, m_std = corr_matrix_from_pairs(pdf, m_list)
        per_task[key] = {
            "mean_matrix": m_mean.to_dict(),
            "std_matrix": m_std.to_dict(),
            "n_dates": len(pdf),
            "mean_date_corr": pdf.drop(columns=["date"]).mean().to_dict() if len(pdf) > 0 else {},
        }

    return {
        "mean_matrix": global_mean,
        "std_matrix": global_std,
        "per_window": per_window,
        "per_seed": per_seed,
        "per_task": per_task,
    }


def _run_correlation(args, timestamp: str):
    """执行相关性分析主流程。"""
    print(f"=== 模型相关性分析 ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Split: {args.split}")
    print(f"Windows: {args.windows}")
    print(f"Seeds: {args.seeds}")

    windows = [int(x.strip()) for x in args.windows.split(",")]
    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    # 加载数据
    print("\n--- 加载数据 ---")
    csv_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not csv_path.exists():
        print(f"[ERROR] 数据文件不存在: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"Data: {len(df)} rows, {df['date'].nunique()} dates, "
          f"{df['stock_code'].nunique()} stocks")

    # 加载序列数据
    sequences_dict = None
    seq_dates_dict = None
    try:
        sequences_dict, seq_dates_dict = _load_sequence_data()
        print(f"Sequences: {len(sequences_dict)} stocks loaded")
    except FileNotFoundError:
        print("[WARN] 序列数据不存在，GRU/AGRU将跳过")

    # 运行分析
    print(f"\n--- 相关性分析 ({len(windows)}窗口 × {len(seeds)}种子 = {len(windows)*len(seeds)}任务) ---")
    results = run_all(df, sequences_dict, seq_dates_dict, windows, seeds, args.split)

    # 打印全局矩阵
    print(f"\n\n{'='*60}")
    print("全局平均相关系数矩阵（Rank Correlation）")
    print(f"{'='*60}")
    print(results["mean_matrix"].to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n标准差:")
    print(results["std_matrix"].to_string(float_format=lambda x: f"{x:.4f}"))

    # 按窗口汇总
    print(f"\n\n{'='*60}")
    print("按窗口汇总")
    print(f"{'='*60}")
    for w_label, w_data in results["per_window"].items():
        print(f"\n--- {w_label} ({w_data['n_tasks']} tasks, {w_data['n_dates']} dates) ---")
        w_mean = pd.DataFrame(w_data["mean"])
        print(w_mean.to_string(float_format=lambda x: f"{x:.4f}"))

    # 保存结果
    save_data = {
        "config": {
            "split": args.split,
            "windows": windows,
            "seeds": seeds,
            "timestamp": timestamp,
        },
        "mean_matrix": results["mean_matrix"].to_dict(),
        "std_matrix": results["std_matrix"].to_dict(),
        "per_window": results["per_window"],
        "per_seed": results["per_seed"],
        "per_task": results["per_task"],
    }

    results_path = LOG_DIR / f"correlation_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n\nResults saved to: {results_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="模型相关性分析")
    parser.add_argument("--no-log", action="store_true", help="不包裹日志")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="评估数据集 (默认test)")
    parser.add_argument("--windows", type=str, default="0,1,2,3,4,5",
                        help="逗号分隔窗口索引")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="逗号分隔种子")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== 日志双路输出 =====
    log_path = None
    if not args.no_log:
        log_dir = r"D:\.shigodo\shigodo\Quantification\wb-remote-monitor\logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"correlation_{timestamp}.log"

    def run():
        _run_correlation(args, timestamp)

    if log_path:
        sys.path.insert(0, r"D:\.shigodo\shigodo\Quantification\wb-remote-monitor")
        from log_to_file import log_writer
        with log_writer(str(log_path), tag="Correlation"):
            run()
        print(f"\n[INFO] Log saved to: {log_path}")
    else:
        run()


if __name__ == "__main__":
    main()
