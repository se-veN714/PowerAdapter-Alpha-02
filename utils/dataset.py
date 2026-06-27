# -*- coding: utf-8 -*-
# @File    : dataset.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了按交易日截面加载的PyTorch数据集类和时间划分函数。

基于SKILL: pytorch-section-dataloader

两种Dataset:
- CrossSectionDataset: MLP/GBDT截面模型，每个样本=(180维截面特征, 标签)
- SequenceDataset: GRU/AGRU时序模型，每个样本=(N,30,6)序列, 标签)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    PROCESSED_DATA_DIR, N_CROSS_SECTION_FEATURES, PRICE_FIELDS,
    SEQUENCE_LENGTH, TRAIN_END, VAL_END, BUFFER_DAYS,
)


class CrossSectionDataset(Dataset):
    """MLP/GBDT的截面数据集。

    每个样本 = 一个交易日截面的所有股票（180维截面特征 + 标签）。
    __getitem__返回(features_tensor, label_tensor)。

    Attributes:
        dates: 排序后的日期列表。
        data: 完整DataFrame。
    """

    def __init__(self, df: pd.DataFrame, label_col: str = "label") -> None:
        """初始化截面数据集。

        Args:
            df: 预处理后的DataFrame，需包含date, stock_code, label和_lag列。
            label_col: 标签列名，默认"label"。GRU/AGRU 使用 "label_vwap10"。
        """
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        self.data = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
        self.dates = sorted(self.data["date"].unique())
        self.label_col = label_col

        # 识别特征列：滞后列 + 静态因子列（排除标签列）
        self.feature_cols = [c for c in self.data.columns if "_lag" in c]
        _label_cols = {"label", "label_close20", "label_vwap10"}
        static_cols = [c for c in self.data.columns
                       if c not in {"date", "stock_code"} | _label_cols
                       and "_lag" not in c
                       and self.data[c].dtype in ("float64", "float32", "int64", "int32")]
        self.feature_cols += static_cols
        if not self.feature_cols:
            print("  [WARN] 未找到_lag列，可能未运行feature_engine")

        # 按日期预分组索引
        self._group_indices: dict[pd.Timestamp, pd.Index] = {}
        for date_val, group in self.data.groupby("date"):
            self._group_indices[pd.Timestamp(date_val)] = group.index

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取第idx个截面的特征和标签。

        Returns:
            (features, labels): features shape=(N, F), labels shape=(N,)
        """
        date_val = self.dates[idx]
        indices = self._group_indices[date_val]
        section = self.data.loc[indices]

        features = section[self.feature_cols].values.astype(np.float32)
        labels = section[self.label_col].values.astype(np.float32)

        return torch.from_numpy(features), torch.from_numpy(labels)

    def get_date(self, idx: int) -> pd.Timestamp:
        """获取第idx个截面对应的日期。"""
        return self.dates[idx]

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


class SequenceDataset(Dataset):
    """GRU/AGRU的序列数据集（高效预构建版本）。

    每个样本 = 一个交易日截面的所有股票（30步×6字段序列 + 标签）。
    在 __init__ 中预构建所有截面张量，避免训练时反复查询 DataFrame。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sequences: dict[str, np.ndarray] | None = None,
        seq_dates: dict[str, np.ndarray] | None = None,
        label_col: str = "label_vwap10",
    ) -> None:
        """初始化序列数据集。

        Args:
            df: 预处理后的DataFrame，需包含date, stock_code, label。
            sequences: 预构建的序列数据 {stock_code: (n_days, seq_length, n_features)}。
            seq_dates: 每条序列对应的目标日期 {stock_code: (n_days,)}。
            label_col: 标签列名，默认"label_vwap10"（论文设定：VWAP 10天）。
        """
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        self.data = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
        self.dates = sorted(self.data["date"].unique())
        self.label_col = label_col
        # 清理序列中的 nan/inf，避免在截面 zscore 和训练时传播
        self.sequences = {
            str(k): np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            for k, v in sequences.items()
        } if sequences else {}
        self.seq_dates = {str(k): v for k, v in seq_dates.items()} if seq_dates else {}

        # 预构建 (stock_code, date) -> sequence position 字典
        self._stock_date_to_pos: dict[tuple[str, pd.Timestamp], int] = {}
        for code, dates in self.seq_dates.items():
            seq_array = self.sequences.get(code)
            if seq_array is None or len(dates) != len(seq_array):
                continue
            for i, d in enumerate(dates):
                date_val = pd.Timestamp(d)
                self._stock_date_to_pos[(code, date_val)] = i

        # 预构建所有截面张量（训练时只需按日期索引）
        self._section_tensors: list[torch.Tensor] = []
        self._section_labels: list[torch.Tensor] = []
        for date_val, group in self.data.groupby("date"):
            codes = group["stock_code"].astype(str).values
            labels = group[self.label_col].values.astype(np.float32)
            seq_list = []
            valid_labels = []
            for code, lab in zip(codes, labels):
                pos = self._stock_date_to_pos.get((code, date_val))
                if pos is not None:
                    seq_list.append(self.sequences[code][pos])
                    valid_labels.append(lab)

            if seq_list:
                self._section_tensors.append(
                    torch.tensor(np.stack(seq_list), dtype=torch.float32)
                )
                self._section_labels.append(
                    torch.tensor(np.array(valid_labels, dtype=np.float32))
                )
            else:
                self._section_tensors.append(
                    torch.zeros((0, SEQUENCE_LENGTH, len(PRICE_FIELDS)), dtype=torch.float32)
                )
                self._section_labels.append(torch.zeros(0, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取第idx个截面的序列特征和标签。"""
        return self._section_tensors[idx], self._section_labels[idx]

    def get_date(self, idx: int) -> pd.Timestamp:
        """获取第idx个截面对应的日期。"""
        return self.dates[idx]


def split_dataset(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
    test_end: str | None = None,
    buffer_days: int = BUFFER_DAYS,
    mode: str = "cross_section",
    sequences: dict[str, np.ndarray] | None = None,
    seq_dates: dict[str, np.ndarray] | None = None,
    label_col: str = "label",
) -> tuple[Dataset, Dataset, Dataset]:
    """按时间切分训练/验证/测试数据集。

    严禁随机split，必须按时间划分以避免未来信息泄露。
    相邻数据集之间剔除buffer_days天防止信息泄露。

    Args:
        df: 预处理后的DataFrame。
        train_end: 训练集截止日期（不含，剔除前buffer_days天）。
        val_end: 验证集截止日期（不含，剔除前buffer_days天）。
        test_end: 测试集截止日期（不含）；若未提供，则取val_end+buffer之后全部数据。
        buffer_days: 相邻剔除天数，默认10。
        mode: "cross_section" → CrossSectionDataset, "sequence" → SequenceDataset。
        sequences: 序列数据（仅mode="sequence"时使用）。
        seq_dates: 序列对应日期（仅mode="sequence"时使用）。

    Returns:
        (train_dataset, val_dataset, test_dataset) 元组。
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    # 训练集: 截止到train_end（剔除最后buffer_days天）
    train_cutoff = train_end_dt - pd.Timedelta(days=buffer_days)
    train_df = df[df["date"] < train_cutoff].copy()

    # 验证集: train_end+buffer 到 val_end-buffer
    val_start = train_end_dt + pd.Timedelta(days=buffer_days)
    val_cutoff = val_end_dt - pd.Timedelta(days=buffer_days)
    val_df = df[(df["date"] >= val_start) & (df["date"] < val_cutoff)].copy()

    # 测试集: val_end+buffer 到 test_end（若未提供则取之后全部）
    test_start = val_end_dt + pd.Timedelta(days=buffer_days)
    if test_end is not None:
        test_end_dt = pd.Timestamp(test_end)
        test_df = df[(df["date"] >= test_start) & (df["date"] < test_end_dt)].copy()
    else:
        test_df = df[df["date"] >= test_start].copy()

    print(f"Train: {len(train_df)} rows, {train_df['date'].nunique()} dates")
    print(f"Val:   {len(val_df)} rows, {val_df['date'].nunique()} dates")
    print(f"Test:  {len(test_df)} rows, {test_df['date'].nunique()} dates")

    if mode == "cross_section":
        return (
            CrossSectionDataset(train_df, label_col=label_col),
            CrossSectionDataset(val_df, label_col=label_col),
            CrossSectionDataset(test_df, label_col=label_col),
        )
    else:  # sequence
        return (
            SequenceDataset(train_df, sequences, seq_dates, label_col=label_col),
            SequenceDataset(val_df, sequences, seq_dates, label_col=label_col),
            SequenceDataset(test_df, sequences, seq_dates, label_col=label_col),
        )


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print("请先运行 feature_engine.py 和 preprocess.py")
    else:
        df = pd.read_csv(processed_path, parse_dates=["date"])
        train_ds, val_ds, test_ds = split_dataset(df, mode="cross_section")

        # 验证单个截面
        x, y = train_ds[0]
        print(f"\nSection 0: features shape={x.shape}, labels shape={y.shape}")
        print(f"Section 0 date: {train_ds.get_date(0)}")
        print(f"Feature dims: {train_ds.n_features} (expected: {N_CROSS_SECTION_FEATURES})")
