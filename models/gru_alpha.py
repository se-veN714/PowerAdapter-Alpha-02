# -*- coding: utf-8 -*-
# @File    : gru_alpha.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了GRU时序量价因子模型的类定义。

论文图9 + 表5:
- 输入: (N, 30, 6) 序列 — 30天×6字段(OHLC+VWAP+VOLUME)
- 结构: 2层GRU(hidden=64, dropout=0.1) → 取最后隐状态 → FC(64, 1)
- 标签: 预测收益zscore
"""

import torch
import torch.nn as nn

from config import (
    N_FEATURES, SEQUENCE_LENGTH,
    GRU_HIDDEN_DIMS, GRU_NUM_LAYERS, GRU_DROPOUT,
)


class GRUAlphaModel(nn.Module):
    """GRU时序量价因子模型（论文图9）。

    Attributes:
        input_size: 每步特征数（6）。
        hidden_size: GRU隐状态维度（64）。
        num_layers: GRU层数（2）。
    """

    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = GRU_HIDDEN_DIMS,
        num_layers: int = GRU_NUM_LAYERS,
        dropout: float = GRU_DROPOUT,
    ) -> None:
        """初始化GRU模型。

        Args:
            input_size: 每步特征数，默认6。
            hidden_size: 隐状态维度，默认64。
            num_layers: GRU层数，默认2。
            dropout: Dropout比例（层间），默认0.1。
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU层: batch_first=True → (batch, seq_len, input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # 输出层: 取最后时间步隐状态 → 预测
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入序列，shape=(N, seq_len, input_size)。

        Returns:
            (prediction, last_hidden):
            - prediction: shape=(N,) 预测zscore
            - last_hidden: shape=(N, hidden_size) 最后时间步隐状态
        """
        # GRU输出: output=(N, seq_len, hidden), h_n=(num_layers, N, hidden)
        gru_out, h_n = self.gru(x)

        # 取最后时间步的隐状态
        last_hidden = gru_out[:, -1, :]  # (N, hidden)

        # 输出预测
        pred = self.output_layer(last_hidden).squeeze(-1)  # (N,)

        return pred, last_hidden

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回预测值。

        Args:
            x: 输入序列，shape=(N, seq_len, input_size)。

        Returns:
            预测值，shape=(N,)。
        """
        pred, _ = self.forward(x)
        return pred


if __name__ == "__main__":
    # 简单测试
    batch_size = 32
    seq_len = SEQUENCE_LENGTH
    n_features = N_FEATURES

    model = GRUAlphaModel()
    x = torch.randn(batch_size, seq_len, n_features)
    pred, hidden = model(x)

    print(f"GRU model:")
    print(f"  Input: {x.shape}")
    print(f"  Hidden: {hidden.shape}")
    print(f"  Prediction: {pred.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
