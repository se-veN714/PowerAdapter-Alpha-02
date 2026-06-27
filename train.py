# -*- coding: utf-8 -*-
# @File    : train.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了模型训练入口。

基于SKILL: time-series-rolling-validation

支持模式:
- single: 单次训练（--model mlp/gbdt/gru/agru）
- rolling: 4模型×6窗口滚动训练

使用方式:
    python train.py --mode single --model mlp
    python train.py --mode rolling
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    DEVICE, RANDOM_SEEDS,
    MLP_LR, MLP_MAX_EPOCHS, MLP_PATIENCE,
    GRU_LR, GRU_MAX_EPOCHS, GRU_PATIENCE, GRU_LOSS_FN, GRU_LABEL_COL,
    AGRU_LOSS_FN, AGRU_LABEL_COL,
    ROLLING_WINDOWS, BUFFER_DAYS,
    LOSS_FN, N_CROSS_SECTION_FEATURES, WEIGHT_DECAY, ORTH_LAMBDA,
)
from losses import get_loss_fn, orthogonal_penalty
from utils.dataset import split_dataset
from utils.metrics import rank_ic


# ===== 训练器 =====

def section_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对序列输入按截面做 zscore 标准化。

    Args:
        x: 输入张量，shape=(N, seq_len, n_features)。
        eps: 防止除零。

    Returns:
        标准化后的张量，shape 与 x 相同。
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    return (x - mean) / (std + eps)


def train_pytorch_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    lr: float,
    max_epochs: int,
    patience: int,
    model_name: str,
    window_idx: int,
    seed: int,
    pbar: tqdm | None = None,  # 外部进度条（用于 rolling 模式）
) -> dict:
    """训练PyTorch模型（MLP/GRU/AGRU）。

    截面DataLoader模式: batch_size=1，每批一个交易日截面。

    Args:
        model: PyTorch模型实例。
        train_loader: 训练DataLoader。
        val_loader: 验证DataLoader。
        loss_fn: 损失函数。
        lr: 学习率。
        max_epochs: 最大epoch数。
        patience: 早停轮数。
        model_name: 模型名称（用于日志）。
        window_idx: 窗口索引。
        seed: 随机种子。
        pbar: 外部tqdm进度条（可选），用于更新描述信息。

    Returns:
        dict: 训练结果（best_val_ic, train_loss_history等）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_val_ic = -float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_ics = []

    # epoch 级进度条
    epoch_pbar = tqdm(range(max_epochs), desc=f"  W{window_idx}/S{seed}", unit="ep", leave=False)
    for epoch in epoch_pbar:
        # ---- Training ----
        model.train()
        epoch_losses = []

        for x_batch, y_batch in train_loader:
            if x_batch.dim() in (3, 4) and x_batch.shape[0] == 1:
                x_batch = x_batch.squeeze(0)
            if y_batch.dim() == 2 and y_batch.shape[0] == 1:
                y_batch = y_batch.squeeze(0)
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if x_batch.shape[0] == 0:
                continue

            # 对序列输入做截面标准化（GRU/AGRU）
            if x_batch.dim() == 3:
                x_batch = section_zscore(x_batch)

            optimizer.zero_grad()

            # MLP返回(pred, hidden), GRU/AGRU也返回(pred, hidden)
            pred, hidden = model(x_batch)
            loss = loss_fn(pred, y_batch) + ORTH_LAMBDA * orthogonal_penalty(hidden)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_losses.append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                if x_batch.dim() in (3, 4) and x_batch.shape[0] == 1:
                    x_batch = x_batch.squeeze(0)
                if y_batch.dim() == 2 and y_batch.shape[0] == 1:
                    y_batch = y_batch.squeeze(0)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                if x_batch.shape[0] == 0:
                    continue

                # 对序列输入做截面标准化（GRU/AGRU）
                if x_batch.dim() == 3:
                    x_batch = section_zscore(x_batch)

                pred, _ = model(x_batch)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        if all_preds:
            val_pred = np.concatenate(all_preds)
            val_label = np.concatenate(all_labels)
            val_ic = rank_ic(val_pred, val_label)
        else:
            val_ic = 0.0

        val_ics.append(val_ic)

        # 更新 epoch 进度条
        epoch_pbar.set_postfix(loss=avg_train_loss, ic=val_ic, best=best_val_ic)
        if pbar is not None:
            pbar.set_postfix({"cur_ic": f"{val_ic:.4f}", "best_ic": f"{best_val_ic:.4f}"})

        # Early stopping based on val_ic
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            ckpt_path = CHECKPOINT_DIR / f"{model_name}_w{window_idx}_s{seed}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_pbar.set_description(f"  W{window_idx}/S{seed} [stop@{epoch+1}]")
                epoch_pbar.close()
                break

    return {
        "best_val_ic": best_val_ic,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_ics": val_ics,
    }


def flatten_cross_section_dataset(dataset):
    """将截面数据集拉平为 (N, F) 和 (N,) 张量，用于 MLP/GBDT 全量训练。

    Args:
        dataset: CrossSectionDataset 实例。

    Returns:
        (X, y): X shape=(N, F), y shape=(N,)。
    """
    x_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        x_list.append(x)
        y_list.append(y)
    X = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y


def train_gbdt_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    model_name: str, window_idx: int, seed: int,
) -> dict:
    """训练LightGBM模型。

    与PyTorch不同，GBDT一次性训练全部数据。

    Args:
        X_train, y_train: 训练数据。
        X_val, y_val: 验证数据。
        model_name: 模型名称。
        window_idx: 窗口索引。
        seed: 随机种子。

    Returns:
        dict: 训练结果。
    """
    from models.gbdt_alpha import GBDTAlphaModel

    np.random.seed(seed)

    model = GBDTAlphaModel(random_state=seed)
    model.fit(X_train, y_train, X_val, y_val)

    # 验证集IC
    pred_val = model.predict(X_val)
    val_ic = rank_ic(pred_val, y_val)

    # Save model
    import joblib
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_w{window_idx}_s{seed}.joblib"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.model, ckpt_path)

    return {
        "best_val_ic": val_ic,
        "n_estimators": model.model.n_estimators,  # 路线A: 固定树数
    }


# ===== 主函数 =====

def train_single(model_name: str, seed: int = 42):
    """单次训练指定模型。

    Args:
        model_name: mlp/gbdt/gru/agru。
        seed: 随机种子。
    """
    print(f"Training {model_name} (seed={seed})...")

    # 加载数据
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print(f"Error: {processed_path} not found. Run feature_engine.py + preprocess.py first.")
        return

    df = pd.read_csv(processed_path, parse_dates=["date"])
    print(f"Data loaded: {len(df)} rows")

    if model_name == "gbdt":
        # 按时间切分
        train_ds, val_ds, test_ds = split_dataset(df)
        # GBDT: 拼接全部训练数据
        X_train_list, y_train_list = [], []
        for i in range(len(train_ds)):
            x, y = train_ds[i]
            X_train_list.append(x.numpy())
            y_train_list.append(y.numpy())
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_val_list, y_val_list = [], []
        for i in range(len(val_ds)):
            x, y = val_ds[i]
            X_val_list.append(x.numpy())
            y_val_list.append(y.numpy())
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)

        result = train_gbdt_model(X_train, y_train, X_val, y_val, model_name, 0, seed)
        print(f"  Result: val_ic={result['best_val_ic']:.6f}, n_estimators={result['n_estimators']}")

    elif model_name == "mlp":
        # 按时间切分
        train_ds, val_ds, test_ds = split_dataset(df)
        # MLP: 拉平为全量样本，使用大 batch 训练以加速
        from models.mlp_alpha import MLPAlphaModel
        from torch.utils.data import TensorDataset

        X_train, y_train = flatten_cross_section_dataset(train_ds)
        X_val, y_val = flatten_cross_section_dataset(val_ds)

        train_flat_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=1024,
            shuffle=True,
            drop_last=False,
        )
        val_flat_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=1024,
            shuffle=False,
            drop_last=False,
        )

        model = MLPAlphaModel(n_factors=N_CROSS_SECTION_FEATURES)
        loss_fn = get_loss_fn(LOSS_FN)()
        result = train_pytorch_model(
            model, train_flat_loader, val_flat_loader, loss_fn,
            MLP_LR, MLP_MAX_EPOCHS, MLP_PATIENCE, model_name, 0, seed,
        )
        print(f"  Result: best_val_ic={result['best_val_ic']:.6f}, best_epoch={result['best_epoch']}")

    else:
        # PyTorch时序模型（GRU/AGRU）
        if model_name == "gru":
            from models.gru_alpha import GRUAlphaModel
            model = GRUAlphaModel()
            lr = GRU_LR
            max_epochs = GRU_MAX_EPOCHS
            patience = GRU_PATIENCE
            loss_fn_name = GRU_LOSS_FN
            label_col = GRU_LABEL_COL
        elif model_name == "agru":
            from models.agru_alpha import AGRUAlphaModel
            model = AGRUAlphaModel()
            lr = GRU_LR
            max_epochs = GRU_MAX_EPOCHS
            patience = GRU_PATIENCE
            loss_fn_name = AGRU_LOSS_FN
            label_col = AGRU_LABEL_COL
        else:
            print(f"Unknown model: {model_name}")
            return

        # 加载序列数据
        seq_path = PROCESSED_DATA_DIR / "sequences.npz"
        seq_dates_path = PROCESSED_DATA_DIR / "seq_dates.npz"
        if not seq_path.exists():
            print(f"Error: {seq_path} not found. Run feature_engine.py to generate sequences.")
            return
        sequences_npz = np.load(seq_path, allow_pickle=True)
        sequences = {k: sequences_npz[k] for k in sequences_npz.files}

        if not seq_dates_path.exists():
            print(f"Error: {seq_dates_path} not found. Run feature_engine.py to generate sequence dates.")
            return
        seq_dates_npz = np.load(seq_dates_path, allow_pickle=True)
        seq_dates = {k: seq_dates_npz[k] for k in seq_dates_npz.files}

        train_ds, val_ds, test_ds = split_dataset(
            df, mode="sequence", sequences=sequences, seq_dates=seq_dates, label_col=label_col,
        )
        loss_fn = get_loss_fn(loss_fn_name)()
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        result = train_pytorch_model(
            model, train_loader, val_loader, loss_fn,
            lr, max_epochs, patience, model_name, 0, seed,
        )
        print(f"  Result: best_val_ic={result['best_val_ic']:.6f}, best_epoch={result['best_epoch']}")
        print(f"  [Paper setting] loss={loss_fn_name}, label={label_col}")



def train_rolling(model_filter: str | None = None):
    """滚动训练：4模型×6窗口（+3种子）。

    Args:
        model_filter: 仅训练指定模型（mlp/gbdt/gru/agru），None=全部。
    """
    model_names = ["mlp", "gbdt", "gru", "agru"] if model_filter is None else [model_filter]
    print("=" * 60)
    print(f"Rolling Training: {len(model_names)} model(s) × {len(ROLLING_WINDOWS)} windows × {len(RANDOM_SEEDS)} seeds")
    print("=" * 60)

    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print(f"Error: {processed_path} not found.")
        return

    df = pd.read_csv(processed_path, parse_dates=["date"])

    # 加载序列数据（GRU/AGRU 使用）
    seq_path = PROCESSED_DATA_DIR / "sequences.npz"
    seq_dates_path = PROCESSED_DATA_DIR / "seq_dates.npz"
    sequences: dict[str, np.ndarray] | None = None
    seq_dates: dict[str, np.ndarray] | None = None
    if seq_path.exists() and seq_dates_path.exists():
        sequences_npz = np.load(seq_path, allow_pickle=True)
        sequences = {k: sequences_npz[k] for k in sequences_npz.files}
        seq_dates_npz = np.load(seq_dates_path, allow_pickle=True)
        seq_dates = {k: seq_dates_npz[k] for k in seq_dates_npz.files}
    else:
        print(f"[WARN] 未找到序列数据，GRU/AGRU 将跳过")

    all_results = {}

    for model_name in model_names:
        print(f"\n{'='*40}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*40}")

        model_results = []

        # 窗口级进度条
        total_tasks = len(ROLLING_WINDOWS) * len(RANDOM_SEEDS)
        master_pbar = tqdm(total=total_tasks, desc=f"  {model_name.upper()}", unit="task", position=0)

        for w, (train_end, val_end, test_end) in enumerate(ROLLING_WINDOWS):
            # 每个窗口分别切分截面/序列数据（所有种子共享数据）
            cs_train_ds, cs_val_ds, cs_test_ds = split_dataset(
                df, train_end=train_end, val_end=val_end, test_end=test_end, mode="cross_section",
            )
            if sequences is not None and seq_dates is not None:
                seq_train_ds, seq_val_ds, seq_test_ds = split_dataset(
                    df, train_end=train_end, val_end=val_end, test_end=test_end, mode="sequence",
                    sequences=sequences, seq_dates=seq_dates, label_col="label_vwap10",
                )
            else:
                seq_train_ds = seq_val_ds = seq_test_ds = None

            seed_results = []
            for seed in RANDOM_SEEDS:
                master_pbar.set_description(f"  {model_name.upper()} W{w} S{seed}")

                if model_name == "gbdt":
                    # GBDT训练：使用截面数据
                    X_train_list, y_train_list = [], []
                    for i in range(len(cs_train_ds)):
                        x, y = cs_train_ds[i]
                        X_train_list.append(x.numpy())
                        y_train_list.append(y.numpy())
                    X_train = np.concatenate(X_train_list, axis=0)
                    y_train = np.concatenate(y_train_list, axis=0)

                    X_val_list, y_val_list = [], []
                    for i in range(len(cs_val_ds)):
                        x, y = cs_val_ds[i]
                        X_val_list.append(x.numpy())
                        y_val_list.append(y.numpy())
                    X_val = np.concatenate(X_val_list, axis=0)
                    y_val = np.concatenate(y_val_list, axis=0)

                    result = train_gbdt_model(
                        X_train, y_train, X_val, y_val,
                        model_name, w, seed,
                    )
                elif model_name == "mlp":
                    # MLP训练：截面数据拉平后大 batch 训练
                    from torch.utils.data import TensorDataset
                    X_train, y_train = flatten_cross_section_dataset(cs_train_ds)
                    X_val, y_val = flatten_cross_section_dataset(cs_val_ds)
                    train_loader = DataLoader(
                        TensorDataset(X_train, y_train),
                        batch_size=1024, shuffle=True, drop_last=False,
                    )
                    val_loader = DataLoader(
                        TensorDataset(X_val, y_val),
                        batch_size=1024, shuffle=False, drop_last=False,
                    )
                    from models.mlp_alpha import MLPAlphaModel
                    model = MLPAlphaModel(n_factors=N_CROSS_SECTION_FEATURES)
                    loss_fn = get_loss_fn(LOSS_FN)()
                    result = train_pytorch_model(
                        model, train_loader, val_loader, loss_fn,
                        MLP_LR, MLP_MAX_EPOCHS, MLP_PATIENCE, model_name, w, seed,
                        pbar=master_pbar,
                    )
                elif model_name in ("gru", "agru"):
                    # GRU/AGRU训练：使用序列数据 + 论文设定（MSE + VWAP 10天）
                    if seq_train_ds is None or seq_val_ds is None:
                        master_pbar.set_description(f"  {model_name.upper()} W{w} [SKIP]")
                        master_pbar.update(1)
                        continue
                    if model_name == "gru":
                        from models.gru_alpha import GRUAlphaModel
                        model = GRUAlphaModel()
                        loss_fn_name = GRU_LOSS_FN
                    else:
                        from models.agru_alpha import AGRUAlphaModel
                        model = AGRUAlphaModel()
                        loss_fn_name = AGRU_LOSS_FN
                    loss_fn = get_loss_fn(loss_fn_name)()
                    train_loader = DataLoader(seq_train_ds, batch_size=1, shuffle=True)
                    val_loader = DataLoader(seq_val_ds, batch_size=1, shuffle=False)
                    result = train_pytorch_model(
                        model, train_loader, val_loader, loss_fn,
                        GRU_LR, GRU_MAX_EPOCHS, GRU_PATIENCE, model_name, w, seed,
                        pbar=master_pbar,
                    )
                else:
                    print(f"Unknown model: {model_name}")
                    master_pbar.update(1)
                    continue

                seed_results.append(result)
                master_pbar.set_postfix(best_ic=f"{result['best_val_ic']:.4f}")
                master_pbar.update(1)

            model_results.append({
                "window": w,
                "train_end": train_end,
                "seeds": seed_results,
            })

        master_pbar.close()
        all_results[model_name] = model_results

    # Save results
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = LOG_DIR / f"rolling_results_{timestamp}.json"

    # Convert to serializable
    serializable = {}
    for model_name, windows in all_results.items():
        serializable[model_name] = []
        for w in windows:
            seed_data = []
            for s in w["seeds"]:
                seed_data.append({
                    "best_val_ic": s["best_val_ic"],
                    "best_epoch": s.get("best_epoch", None),
                })
            serializable[model_name].append({
                "window": w["window"],
                "train_end": w["train_end"],
                "seeds": seed_data,
            })

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project-Beta Training")
    parser.add_argument("--mode", choices=["single", "rolling"], default="single", help="Training mode")
    parser.add_argument("--model", choices=["mlp", "gbdt", "gru", "agru"], default=None,
                       help="Model filter (rolling: None=all, single: defaults to mlp)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-log", action="store_true", help="Disable dual-output logging")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ===== 日志双路输出 =====
    log_path = None
    if not args.no_log:
        log_dir = r"D:\.shigodo\shigodo\Quantification\wb-remote-monitor\logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"training_{ts}.log"

    if log_path:
        sys.path.insert(0, r"D:\.shigodo\shigodo\Quantification\wb-remote-monitor")
        from log_to_file import log_writer
        with log_writer(str(log_path)):
            if args.mode == "single":
                train_single(args.model or "mlp", args.seed)
            else:
                train_rolling(model_filter=args.model)
    else:
        if args.mode == "single":
            train_single(args.model or "mlp", args.seed)
        else:
            train_rolling(model_filter=args.model)
