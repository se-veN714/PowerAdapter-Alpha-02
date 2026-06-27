# -*- coding: utf-8 -*-
# @File    : mlp_alpha.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了MLP截面量价因子模型的类定义。

基于Project-Alpha MLP架构，适配论文二参数:
- 输入: 180维截面特征 (6字段×30天滞后)
- 结构: FC(180,64)→BN→LeakyReLU→Dropout→FC(64,32)→BN→LeakyReLU→Dropout→FC(32,1)
- 激活函数: LeakyReLU(0.01)（Project-Alpha最优，允许负值通过）
- 末层无激活函数（标签zscore有正负）
"""

import torch
import torch.nn as nn

from config import MLP_HIDDEN_DIMS, MLP_DROPOUT


class MLPAlphaModel(nn.Module):
    """MLP截面量价因子模型（论文图2）。

    输入180维展开特征，输出截面zscore预测。
    forward返回(prediction, last_hidden)用于正交惩罚（如需）。

    Attributes:
        n_factors: 输入特征维度（180）。
        hidden_dims: 隐藏层维度序列（默认3层128维）。
    """

    def __init__(
        self,
        n_factors: int,
        hidden_dims: tuple = MLP_HIDDEN_DIMS,
        dropout: float = MLP_DROPOUT,
    ) -> None:
        """初始化MLP模型。

        Args:
            n_factors: 输入特征维度（默认180）。
            hidden_dims: 隐藏层维度序列，默认(128, 128, 128)。
            dropout: Dropout比例，默认0.05。
        """
        super().__init__()
        self.n_factors = n_factors
        self.hidden_dims = hidden_dims

        # 构建隐藏层: Linear → BN → LeakyReLU → Dropout
        layers = []
        in_dim = n_factors
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)

        # 输出层: 末层无激活函数（标签zscore有正负）
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入特征，shape=(N, n_factors)。

        Returns:
            (prediction, last_hidden): prediction shape=(N,), last_hidden shape=(N, hidden_dim)。
        """
        hidden = self.hidden_layers(x)
        pred = self.output_layer(hidden).squeeze(-1)
        return pred, hidden

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回预测值（评估时使用）。

        Args:
            x: 输入特征，shape=(N, n_factors)。

        Returns:
            预测值，shape=(N,)。
        """
        pred, _ = self.forward(x)
        return pred
