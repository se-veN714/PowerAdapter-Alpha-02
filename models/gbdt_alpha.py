# -*- coding: utf-8 -*-
# @File    : gbdt_alpha.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)

"""本模块提供了LightGBM截面量价因子模型的类定义。

非PyTorch模型，使用lightgbm.LGBMRegressor。
输入: 180维截面特征（6字段×30天滞后展开）。

路线B论文表4参数:
- learning_rate=0.1, max_depth=7, num_leaves=127
- min_data_in_leaf=512, feature_fraction=0.7, bagging_fraction=0.7
- n_estimators=500 (180维量价特征需更多树)
"""

import numpy as np
import lightgbm as lgbm

from config import (
    LGBM_LR, LGBM_MAX_DEPTH, LGBM_NUM_LEAVES,
    LGBM_MIN_DATA_IN_LEAF, LGBM_FEATURE_FRACTION,
    LGBM_BAGGING_FRACTION, LGBM_PATIENCE,
)


class GBDTAlphaModel:
    """LightGBM截面量价因子模型（论文表4）。

    GBDT训练模式与PyTorch不同:
    - PyTorch模型逐截面batch训练
    - LightGBM一次性训练全部数据（非逐截面）
    - 需在train.py中区分两种训练模式

    Attributes:
        model: lightgbm.LGBMRegressor实例。
        feature_names: 特征名列（可选）。
    """

    def __init__(
        self,
        learning_rate: float = LGBM_LR,
        max_depth: int = LGBM_MAX_DEPTH,
        num_leaves: int = LGBM_NUM_LEAVES,
        min_data_in_leaf: int = LGBM_MIN_DATA_IN_LEAF,
        feature_fraction: float = LGBM_FEATURE_FRACTION,
        bagging_fraction: float = LGBM_BAGGING_FRACTION,
        early_stopping_rounds: int = LGBM_PATIENCE,
        random_state: int = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        """初始化GBDT模型。

        Args:
            learning_rate: 学习率，默认0.01。
            max_depth: 树最大深度，默认64。
            num_leaves: 叶节点数，默认512。
            min_data_in_leaf: 叶最小样本数，默认512。
            feature_fraction: 特征采样率，默认0.7。
            bagging_fraction: 样本采样率，默认0.7。
            early_stopping_rounds: 早停轮数，默认50。
            random_state: 随机种子。
            feature_names: 特征名称列表。
        """
        self.model = lgbm.LGBMRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_data_in_leaf,
            colsample_bytree=feature_fraction,
            subsample=bagging_fraction,
            subsample_freq=1,
            random_state=random_state,
            n_estimators=500,  # 路线B: 180维量价特征需更多树（论文设定）
            verbose=-1,
            n_jobs=-1,
        )
        self._early_stopping_rounds = early_stopping_rounds
        self.feature_names = feature_names

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """训练模型。

        一次性训练全部截面数据，非逐截面训练。

        Args:
            X: 训练特征矩阵 (n_samples, n_features)。
            y: 训练标签数组 (n_samples,)。
            X_val: 验证特征矩阵（可选）。
            y_val: 验证标签数组（可选）。
        """
        eval_set = [(X_val, y_val)] if X_val is not None else None
        eval_metric = "l2"  # 路线B: 启用MSE早停（防止180维过拟合）

        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric=eval_metric,
            callbacks=[
                lgbm.early_stopping(self._early_stopping_rounds, verbose=False),
                lgbm.log_evaluation(0),
            ] if eval_set else [],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测。

        Args:
            X: 特征矩阵 (n_samples, n_features)。

        Returns:
            预测值数组 (n_samples,)。
        """
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要度（split次数）。

        Returns:
            特征重要度数组。
        """
        return self.model.feature_importances_


if __name__ == "__main__":
    # 简单测试
    np.random.seed(42)
    X = np.random.randn(1000, 180).astype(np.float32)
    y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(1000)

    model = GBDTAlphaModel(feature_names=[f"f_{i}" for i in range(180)])
    model.fit(X[:800], y[:800], X[800:], y[800:])

    pred = model.predict(X[800:])
    ic = np.corrcoef(pred, y[800:])[0, 1]
    print(f"GBDT test: IC={ic:.4f}, n_estimators={model.model.best_iteration_}")
