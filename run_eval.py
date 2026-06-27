# -*- coding: utf-8 -*-
# @File    : run_eval.py
# @Time    : 2026/06/26
# @Project : Project-Alpha-02 (路线B)

"""路线B 滚动评估 + 分组回测。

遍历所有窗口×种子，对4个模型和3种集成进行完整评估：
- RankIC / ICIR / WinRate
- 20组分组回测
- 生成图表

使用方式:
    python run_eval.py
    python run_eval.py --no-log
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    N_GROUPS, RANDOM_SEEDS, ROLLING_WINDOWS, ICIR_WINDOW,
)
from run_ensemble import generate_predictions, _load_sequence_data, build_ensemble_daily, compute_model_ic_history
from utils.metrics import calc_ic_series, ic_summary, group_return

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODELS = ["mlp", "gbdt", "gru", "agru"]
ENSEMBLE_TYPES = ["cs_ensemble", "seq_ensemble", "full_ensemble"]
FIG_DIR = LOG_DIR / "figures"


def evaluate_all(
    df: pd.DataFrame,
    sequences: dict,
    seq_dates: dict,
    windows: list[int],
    seeds: list[int],
) -> dict:
    """滚动评估所有模型和集成。"""
    all_results = {}
    total = len(windows) * len(seeds)

    for task_idx, w in enumerate(windows):
        for s in seeds:
            key = f"w{w}_s{s}"
            train_end, val_end, test_end = ROLLING_WINDOWS[w]
            print(f"\n[{(task_idx * len(seeds)) + seeds.index(s) + 1}/{total}] W{w} S{s} | {train_end} → {val_end} → {test_end}")

            # --- 生成预测 ---
            val_preds = {}
            test_preds = {}
            for model in MODELS:
                try:
                    val_preds[model] = generate_predictions(
                        model, w, s, df, sequences=sequences, seq_dates=seq_dates, target_split="val")
                    test_preds[model] = generate_predictions(
                        model, w, s, df, sequences=sequences, seq_dates=seq_dates, target_split="test")
                except Exception as e:
                    print(f"  {model}: SKIP ({e})")

            # --- IC histories (val) ---
            ic_histories = {}
            for model, pdf in val_preds.items():
                ic_s = calc_ic_series(pdf["prediction"], pdf["label"], pdf["date"])
                ic_histories[model] = ic_s

            # --- 单模型测试集评估 ---
            single = {}
            for model, pdf in test_preds.items():
                ic_s = calc_ic_series(pdf["prediction"], pdf["label"], pdf["date"])
                stats = ic_summary(ic_s)
                gr = group_return(pdf, n_groups=N_GROUPS)
                single[model] = {
                    "ic_stats": stats,
                    "group_return": gr.to_dict(orient="records"),
                    "n_dates": int(pdf["date"].nunique()),
                    "n_preds": len(pdf),
                }

            # --- 集成 ---
            ensembles = {}
            # CS
            if "mlp" in test_preds and "gbdt" in test_preds:
                cs_test = {"mlp": test_preds["mlp"], "gbdt": test_preds["gbdt"]}
                cs_ic = {"mlp": ic_histories["mlp"], "gbdt": ic_histories["gbdt"]}
                cs_df = build_ensemble_daily(cs_test, cs_ic)
                if len(cs_df) > 0:
                    ensembles["cs_ensemble"] = eval_ensemble(cs_df)
            # SEQ
            if "gru" in test_preds and "agru" in test_preds:
                seq_test = {"gru": test_preds["gru"], "agru": test_preds["agru"]}
                seq_ic = {"gru": ic_histories["gru"], "agru": ic_histories["agru"]}
                seq_df = build_ensemble_daily(seq_test, seq_ic)
                if len(seq_df) > 0:
                    ensembles["seq_ensemble"] = eval_ensemble(seq_df)
            # Full
            if len(test_preds) >= 3:
                full_df = build_ensemble_daily(test_preds, ic_histories)
                if len(full_df) > 0:
                    ensembles["full_ensemble"] = eval_ensemble(full_df)

            all_results[key] = {
                "single": single,
                "ensemble": ensembles,
                "ic_histories": {m: {str(k): v for k, v in ic.to_dict().items()} for m, ic in ic_histories.items()},
            }

    return all_results


def eval_ensemble(df: pd.DataFrame) -> dict:
    """评估单个集成DataFrame。"""
    ic_s = calc_ic_series(df["prediction"], df["label"], df["date"])
    stats = ic_summary(ic_s)
    gr = group_return(df, n_groups=N_GROUPS)
    return {
        "ic_stats": stats,
        "group_return": gr.to_dict(orient="records"),
        "n_dates": int(df["date"].nunique()),
    }


# ============================================================
# 图表
# ============================================================

def plot_rolling_ic(all_results: dict, save_path: str):
    """绘制4模型滚动窗口 Val IC 热力图。"""
    models = MODELS
    windows = sorted(set(int(k.split("_")[0][1:]) for k in all_results))
    seeds = sorted(set(int(k.split("_s")[1]) for k in all_results))

    data = {m: np.zeros((len(windows), len(seeds))) for m in models}
    for key, result in all_results.items():
        w = int(key.split("_")[0][1:])
        s = int(key.split("_s")[1])
        wi = windows.index(w)
        si = seeds.index(s)
        for m in models:
            if m in result.get("single", {}):
                data[m][wi, si] = result["single"][m]["ic_stats"]["rank_ic_mean"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, m in zip(axes.flat, models):
        im = ax.imshow(data[m], cmap="RdYlGn", aspect="auto", vmin=-0.08, vmax=0.08)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"S{s}" for s in seeds])
        ax.set_yticks(range(len(windows)))
        ax.set_yticklabels([f"W{w}" for w in windows])
        ax.set_title(m.upper())
        for i in range(len(windows)):
            for j in range(len(seeds)):
                ax.text(j, i, f"{data[m][i,j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Rank IC")
    fig.suptitle("Route B: Rolling Test RankIC Heatmap (4 models × 6 windows × 3 seeds)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {save_path}")


def plot_ensemble_comparison(all_results: dict, save_path: str):
    """集成 vs 单模型对比柱状图。"""
    # 汇总
    model_ics = {m: [] for m in MODELS}
    ensemble_ics = {e: [] for e in ENSEMBLE_TYPES}
    for result in all_results.values():
        for m in MODELS:
            if m in result.get("single", {}):
                model_ics[m].append(result["single"][m]["ic_stats"]["rank_ic_mean"])
        for e in ENSEMBLE_TYPES:
            if e in result.get("ensemble", {}):
                ensemble_ics[e].append(result["ensemble"][e]["ic_stats"]["rank_ic_mean"])

    categories = MODELS + ["CS-E", "SEQ-E", "FULL-E"]
    means = []
    stds = []
    for m in MODELS:
        vals = model_ics[m]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)
    for e in ENSEMBLE_TYPES:
        vals = ensemble_ics[e]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in means]
    bars = ax.bar(categories, means, yerr=stds, color=colors, capsize=5, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Test RankIC")
    ax.set_title("Route B: Model vs Ensemble Test IC Comparison")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.008,
                f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {save_path}")


def plot_group_backtest(all_results: dict, save_path: str):
    """20组分组回测（全部任务平均）。"""
    # 汇总每个模型的所有 group_return
    model_groups = {m: [] for m in MODELS}
    ensemble_groups = {e: [] for e in ENSEMBLE_TYPES}

    for result in all_results.values():
        for m in MODELS:
            if m in result.get("single", {}) and "group_return" in result["single"][m]:
                gr_data = result["single"][m]["group_return"]
                if gr_data:
                    model_groups[m].append(pd.DataFrame(gr_data))
        for e in ENSEMBLE_TYPES:
            if e in result.get("ensemble", {}) and "group_return" in result["ensemble"][e]:
                gr_data = result["ensemble"][e]["group_return"]
                if gr_data:
                    ensemble_groups[e].append(pd.DataFrame(gr_data))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左: 单模型
    ax = axes[0]
    for m in MODELS:
        if model_groups[m]:
            all_gr = pd.concat(model_groups[m])
            all_gr = all_gr[all_gr["group"] != "long_short"]
            avg = all_gr.groupby("group")["mean_return"].mean()
            ax.plot(avg.index, avg.values, "-o", label=m.upper(), markersize=4)
    ax.set_xlabel("Group (1=Lowest, 20=Highest)")
    ax.set_ylabel("Mean Return")
    ax.set_title("Single Models: 20-Group Backtest")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    # 右: 集成
    ax = axes[1]
    for e in ENSEMBLE_TYPES:
        if ensemble_groups[e]:
            all_gr = pd.concat(ensemble_groups[e])
            all_gr = all_gr[all_gr["group"] != "long_short"]
            avg = all_gr.groupby("group")["mean_return"].mean()
            label_map = {"cs_ensemble": "CS-E", "seq_ensemble": "SEQ-E", "full_ensemble": "FULL-E"}
            ax.plot(avg.index, avg.values, "-o", label=label_map.get(e, e), markersize=4)
    ax.set_xlabel("Group (1=Lowest, 20=Highest)")
    ax.set_ylabel("Mean Return")
    ax.set_title("Ensembles: 20-Group Backtest")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Route B: 20-Group Backtest (all windows × seeds averaged)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {save_path}")


def plot_correlation_heatmap(corr_path: str, save_path: str):
    """加载相关性结果绘制热力图。"""
    if not Path(corr_path).exists():
        print(f"  [SKIP] Correlation file not found: {corr_path}")
        return
    with open(corr_path) as f:
        data = json.load(f)
    mean_mat = pd.DataFrame(data["mean_matrix"])
    # 只保留存在的模型
    models_order = [m for m in MODELS if m in mean_mat.columns]
    mean_mat = mean_mat.loc[models_order, models_order]
    std_mat = pd.DataFrame(data["std_matrix"]).loc[models_order, models_order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(mean_mat.values, cmap="coolwarm", vmin=-0.5, vmax=1.0, aspect="auto")
    axes[0].set_xticks(range(len(models_order)))
    axes[0].set_xticklabels([m.upper() for m in models_order])
    axes[0].set_yticks(range(len(models_order)))
    axes[0].set_yticklabels([m.upper() for m in models_order])
    axes[0].set_title("Mean Cross-Section Rank Correlation")
    for i in range(len(models_order)):
        for j in range(len(models_order)):
            axes[0].text(j, i, f"{mean_mat.iloc[i,j]:.3f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(std_mat.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    axes[1].set_xticks(range(len(models_order)))
    axes[1].set_xticklabels([m.upper() for m in models_order])
    axes[1].set_yticks(range(len(models_order)))
    axes[1].set_yticklabels([m.upper() for m in models_order])
    axes[1].set_title("Std of Correlation")
    for i in range(len(models_order)):
        for j in range(len(models_order)):
            axes[1].text(j, i, f"{std_mat.iloc[i,j]:.3f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    fig.suptitle("Route B: Model Cross-Section Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {save_path}")


# ============================================================
# 主流程
# ============================================================

def _run(args, timestamp: str):
    print("=" * 60)
    print("  Route B: Rolling Evaluation + Group Backtest")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    windows = [int(x) for x in args.windows.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    # 加载数据
    csv_path = PROCESSED_DATA_DIR / "processed_data.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"\nData: {len(df)} rows, {df['date'].nunique()} dates")

    sequences, seq_dates = _load_sequence_data()

    # 评估
    results = evaluate_all(df, sequences, seq_dates, windows, seeds)

    # 汇总
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    model_ics = {m: [] for m in MODELS}
    model_icirs = {m: [] for m in MODELS}
    model_wr = {m: [] for m in MODELS}
    ensemble_metrics = {e: {"ic": [], "icir": [], "wr": []} for e in ENSEMBLE_TYPES}

    for result in results.values():
        for m in MODELS:
            if m in result.get("single", {}):
                s = result["single"][m]["ic_stats"]
                model_ics[m].append(s["rank_ic_mean"])
                model_icirs[m].append(s["icir"])
                model_wr[m].append(s["ic_win_rate"])
        for e in ENSEMBLE_TYPES:
            if e in result.get("ensemble", {}):
                s = result["ensemble"][e]["ic_stats"]
                ensemble_metrics[e]["ic"].append(s["rank_ic_mean"])
                ensemble_metrics[e]["icir"].append(s["icir"])
                ensemble_metrics[e]["wr"].append(s["ic_win_rate"])

    summary = {}
    print("\n--- Single Models ---")
    for m in MODELS:
        if model_ics[m]:
            summary[m] = {
                "mean_ic": round(np.mean(model_ics[m]), 6),
                "std_ic": round(np.std(model_ics[m]), 6),
                "mean_icir": round(np.mean(model_icirs[m]), 6),
                "mean_win_rate": round(np.mean(model_wr[m]), 6),
                "n_tasks": len(model_ics[m]),
            }
            print(f"  {m.upper():5s}: IC={np.mean(model_ics[m]):.4f} ± {np.std(model_ics[m]):.4f}, "
                  f"ICIR={np.mean(model_icirs[m]):.4f}, WR={np.mean(model_wr[m]):.2%}")

    print("\n--- Ensembles ---")
    label_map = {"cs_ensemble": "CS-E", "seq_ensemble": "SEQ-E", "full_ensemble": "FULL-E"}
    for e in ENSEMBLE_TYPES:
        if ensemble_metrics[e]["ic"]:
            summary[e] = {
                "mean_ic": round(np.mean(ensemble_metrics[e]["ic"]), 6),
                "std_ic": round(np.std(ensemble_metrics[e]["ic"]), 6),
                "mean_icir": round(np.mean(ensemble_metrics[e]["icir"]), 6),
                "mean_win_rate": round(np.mean(ensemble_metrics[e]["wr"]), 6),
                "n_tasks": len(ensemble_metrics[e]["ic"]),
            }
            print(f"  {label_map.get(e, e):6s}: IC={np.mean(ensemble_metrics[e]['ic']):.4f} ± "
                  f"{np.std(ensemble_metrics[e]['ic']):.4f}, ICIR={np.mean(ensemble_metrics[e]['icir']):.4f}, "
                  f"WR={np.mean(ensemble_metrics[e]['wr']):.2%}")

    # 保存结果
    save_data = {
        "config": {"windows": windows, "seeds": seeds, "timestamp": timestamp},
        "per_task": results,
        "summary": summary,
    }
    results_path = LOG_DIR / f"eval_rolling_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved: {results_path}")

    # 生成图表
    print(f"\n--- Generating Charts ---")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_rolling_ic(results, str(FIG_DIR / f"routeB_rolling_ic_{timestamp}.png"))
    plot_ensemble_comparison(results, str(FIG_DIR / f"routeB_ensemble_comp_{timestamp}.png"))
    plot_group_backtest(results, str(FIG_DIR / f"routeB_group_backtest_{timestamp}.png"))

    # 相关性热力图（加载已有结果）
    corr_files = sorted(LOG_DIR.glob("correlation_20260626_*.json"), reverse=True)
    if corr_files:
        plot_correlation_heatmap(str(corr_files[0]), str(FIG_DIR / f"routeB_correlation_{timestamp}.png"))

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--windows", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = None
    if not args.no_log:
        log_dir = r"D:\.shigodo\shigodo\Quantification\wb-remote-monitor\logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"eval_{timestamp}.log"

    if log_path:
        sys.path.insert(0, r"D:\.shigodo\shigodo\Quantification\wb-remote-monitor")
        from log_to_file import log_writer
        with log_writer(str(log_path), tag="Eval"):
            _run(args, timestamp)
        print(f"\n[INFO] Log saved to: {log_path}")
    else:
        _run(args, timestamp)


if __name__ == "__main__":
    main()
