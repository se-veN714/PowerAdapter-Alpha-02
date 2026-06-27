# -*- coding: utf-8 -*-
# @File    : losses.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了损失函数定义。

论文二只用MSE loss（见表3/表4/表5），与Project-Alpha不同。
保留IC/CCC/正交惩罚以备后续实验对比。
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """均方误差损失。

    论文二默认损失函数，稳定可靠。

    Args:
        pred: 模型预测值，shape=(N,)。
        target: 真实标签值，shape=(N,)。

    Returns:
        MSE损失标量。
    """
    return F.mse_loss(pred, target)


def ic_loss(pred: Tensor, target: Tensor) -> Tensor:
    """IC损失 = -Pearson相关系数。

    直接优化排序能力，非论文默认但可用于对比实验。

    Args:
        pred: 模型预测值，shape=(N,)。
        target: 真实标签值，shape=(N,)。

    Returns:
        -Pearson相关系数标量。
    """
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()

    pred_std = pred_centered.std()
    target_std = target_centered.std()

    eps = 1e-8
    if pred_std < eps or target_std < eps:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    correlation = (pred_centered * target_centered).mean() / (pred_std * target_std + eps)
    return -correlation


def ccc_loss(pred: Tensor, target: Tensor) -> Tensor:
    """CCC一致性相关系数损失。

    CCC = 2 * rho * sigma_pred * sigma_target / (sigma_pred^2 + sigma_target^2 + (mu_pred - mu_target)^2)
    结合MSE和IC的优点。

    Args:
        pred: 模型预测值，shape=(N,)。
        target: 真实标签值，shape=(N,)。

    Returns:
        -CCC标量。
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var()
    target_var = target.var()

    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    pred_std = torch.sqrt(pred_var + 1e-8)
    target_std = torch.sqrt(target_var + 1e-8)
    correlation = covariance / (pred_std * target_std + 1e-8)

    numerator = 2 * correlation * pred_std * target_std
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2

    ccc = numerator / (denominator + 1e-8)
    return -ccc


# 损失函数注册表
LOSS_FUNCTIONS: dict[str, type] = {
    "mse": type("MSELossFn", (), {"__call__": staticmethod(mse_loss)}),
    "ic": type("ICLossFn", (), {"__call__": staticmethod(ic_loss)}),
    "ccc": type("CCCLossFn", (), {"__call__": staticmethod(ccc_loss)}),
}


def get_loss_fn(name: str) -> type:
    """按名称获取损失函数。

    Args:
        name: 损失函数名称，"mse"/"ic"/"ccc"。

    Returns:
        对应的损失函数类。

    Raises:
        ValueError: 不支持的损失函数名称。
    """
    if name not in LOSS_FUNCTIONS:
        msg = f"不支持的损失函数: {name}，可选: {list(LOSS_FUNCTIONS.keys())}"
        raise ValueError(msg)
    return LOSS_FUNCTIONS[name]


def orthogonal_penalty(hidden: Tensor) -> Tensor:
    """正交惩罚：强制隐因子彼此不相关，增加信息多样性。

    论文核心创新（Report R4）：惩罚隐因子间的相关系数矩阵偏离单位矩阵。
    loss = ||corr(hidden) - I||² / d，d为隐因子维度。

    Args:
        hidden: 隐层输出，shape=(N, d)，d为隐藏层维度。

    Returns:
        正交惩罚标量。
    """
    d = hidden.shape[-1]
    if d < 2:
        return torch.tensor(0.0, device=hidden.device)

    # 标准化每个隐因子（Column-wise zscore）
    hidden_centered = hidden - hidden.mean(dim=0, keepdim=True)
    hidden_std = hidden_centered.std(dim=0, keepdim=True) + 1e-8
    hidden_norm = hidden_centered / hidden_std

    # 相关系数矩阵: d×d
    corr = (hidden_norm.T @ hidden_norm) / hidden_norm.shape[0]

    # 惩罚偏离单位矩阵（off-diagonal）
    diag = torch.eye(d, device=hidden.device)
    return ((corr - diag) ** 2).mean()
