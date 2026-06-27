# -*- coding: utf-8 -*-
# @File    : agru_alpha.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了AGRU（GRU + Self-Attention）时序量价因子模型的类定义。

论文图10: AGRU = GRU + Self-Attention
1. GRU处理序列 → 得到30个隐状态 h_{t-n},...,h_{t-1}
2. 对隐状态计算Self-Attention分数
3. Attention输出与最后隐状态拼接
4. FC(128, 1)  → 预测收益zscore

注意: 论文发现AGRU表现未明显优于GRU（RankIC 10.67% vs 11.27%），
     先跳过高优先级，验证核心管线后再加。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_FEATURES,
    GRU_HIDDEN_DIMS, GRU_NUM_LAYERS, GRU_DROPOUT,
)


class SelfAttention(nn.Module):
    """Self-Attention模块（对隐状态序列计算注意力分数）。"""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = hidden_size ** 0.5

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """计算Self-Attention输出。

        Args:
            h: GRU隐状态序列，shape=(N, seq_len, hidden_size)。

        Returns:
            Attention加权输出，shape=(N, hidden_size)。
        """
        # Scaled Dot-Product Attention
        scores = torch.bmm(h, h.transpose(1, 2)) / self.scale  # (N, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)  # (N, seq_len, seq_len)
        attn_output = torch.bmm(attn_weights, h)  # (N, seq_len, hidden_size)

        # 取平均作为attention表示
        return attn_output.mean(dim=1)  # (N, hidden_size)


class AGRUAlphaModel(nn.Module):
    """AGRU模型（论文图10）= GRU + Self-Attention。

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
        """初始化AGRU模型。

        Args:
            input_size: 每步特征数，默认6。
            hidden_size: 隐状态维度，默认64。
            num_layers: GRU层数，默认2。
            dropout: Dropout比例，默认0.1。
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Self-Attention
        self.attention = SelfAttention(hidden_size)

        # 拼接: last_hidden + attention_output → FC(2*hidden, 1)
        self.output_layer = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入序列，shape=(N, seq_len, input_size)。

        Returns:
            (prediction, last_hidden):
            - prediction: shape=(N,) 预测zscore
            - last_hidden: shape=(N, hidden_size) GRU最后隐状态
        """
        # GRU
        gru_out, h_n = self.gru(x)  # gru_out: (N, seq_len, hidden)

        # Self-Attention
        attn_out = self.attention(gru_out)  # (N, hidden)

        # 拼接最后隐状态 + attention输出
        last_hidden = gru_out[:, -1, :]  # (N, hidden)
        combined = torch.cat([last_hidden, attn_out], dim=-1)  # (N, 2*hidden)

        # 输出预测
        pred = self.output_layer(combined).squeeze(-1)  # (N,)

        return pred, last_hidden

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回预测值。"""
        pred, _ = self.forward(x)
        return pred


if __name__ == "__main__":
    batch_size = 32
    seq_len = 30
    n_features = 6

    model = AGRUAlphaModel()
    x = torch.randn(batch_size, seq_len, n_features)
    pred, hidden = model(x)

    print(f"AGRU model:")
    print(f"  Input: {x.shape}")
    print(f"  Hidden: {hidden.shape}")
    print(f"  Prediction: {pred.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
