# -*- coding: utf-8 -*-
# @File    : config.py
# @Time    : 2026/06/25
# @Project : Project-Beta (Alpha-02)
# 复现招商证券《多模型集成量价Alpha策略》

"""本模块提供了项目全局参数配置的常量和变量。

所有可调参数集中管理，禁止硬编码散落各文件。
"""

from pathlib import Path
from typing import Final

try:
    import torch
    _CUDA_AVAILABLE: Final = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE: Final = False

# ===== 路径配置 =====
PROJECT_ROOT: Final = Path(__file__).resolve().parent
RAW_DATA_DIR: Final = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR: Final = PROJECT_ROOT / "checkpoints"
LOG_DIR: Final = PROJECT_ROOT / "logs"

# ===== Project-Alpha 数据路径（复用原始OHLCV数据） =====
ALPHA_RAW_DIR: Final = PROJECT_ROOT.parent / "Project-Alpha" / "data" / "raw"
ALPHA_STOCK_DATA: Final = ALPHA_RAW_DIR / "stock_data.csv"
ALPHA_FACTOR_DATA: Final = ALPHA_RAW_DIR / "stock_data_with_factors.csv"  # Tier3: 含17个因子

# ===== 股票池（与Project-Alpha相同，50只，覆盖主要行业） =====
STOCK_POOL: Final = [
    # ---- 银行 (6) ----
    "600036", "601398", "601288", "601166", "600000", "601328",
    # ---- 保险/券商 (4) ----
    "601318", "600030", "601601", "601688",
    # ---- 食品饮料 (5) ----
    "600519", "000858", "000568", "000333", "600887",
    # ---- 家电 (1) ----
    "000651",
    # ---- 科技/制造/电子 (6) ----
    "002415", "300750", "002475", "002230", "600588", "002049",
    # ---- 医药 (5) ----
    "600276", "000538", "300760", "600196", "002007",
    # ---- 地产/建材 (3) ----
    "000002", "600048", "600585",
    # ---- 钢铁/有色 (3) ----
    "600019", "601899", "600362",
    # ---- 化工 (3) ----
    "600309", "600426", "002493",
    # ---- 电力/公用 (3) ----
    "600900", "601985", "600025",
    # ---- 建筑 (2) ----
    "601668", "601186",
    # ---- 通信/传媒 (2) ----
    "600050", "000063",
    # ---- 石油/煤炭 (3) ----
    "601857", "601088", "601225",
    # ---- 新能源 (2) ----
    "601012", "002129",
    # ---- 汽车 (2) ----
    "600104", "002594",
]

# ===== 时间范围 =====
DATE_RANGE: Final = ("2018-01-01", "2023-12-31")

# ===== 量价数据字段 =====
PRICE_FIELDS: Final = ["open", "high", "low", "close", "vwap", "volume"]

# ===== Tier2: 量价衍生特征（从原始数据计算） =====
DERIVED_FIELDS: Final = [
    "return_1d",        # 日收益率
    "return_5d",        # 5日收益率
    "return_20d",       # 20日收益率
    "volatility_10d",   # 10日波动率
    "volume_ratio_5d",  # 量比（volume / 5日均量）
    "intraday_spread",  # 日内振幅 (high-low)/close
    "close_open_ratio", # 隔夜跳空 close/open
]

# ===== 路线B: 量价原始特征展开（论文二核心设定） =====
# 6个日线量价字段 × 30天滞后展开 → 180维截面特征（MLP/GBDT输入）
# OHLC+VWAP+VOLUME = 6字段，对应GRU 30步×6维序列
EXPAND_FIELDS: Final = PRICE_FIELDS  # ["open","high","low","close","vwap","volume"]
STATIC_FIELDS: Final = []            # 路线B无静态因子（量价路线）
N_CROSS_SECTION_FEATURES: Final = 180  # 6字段 × 30天滞后 = 180维

# 保留路线A因子定义（供参考/回退）
ALPHA_FACTOR_11: Final = [
    "ep", "bp", "roe", "roe_growth", "profit_growth", "revenue_growth",
    "asset_turnover", "turnover_rate", "amplitude", "momentum_20", "reversal_5",
]
ALPHA_FACTOR_DIRECTION: Final = {
    "ep": 1, "bp": 1, "roe": 1, "roe_growth": 1,
    "profit_growth": 1, "revenue_growth": 1, "asset_turnover": 1,
    "turnover_rate": -1, "amplitude": -1, "momentum_20": 1, "reversal_5": -1,
}

# ===== Tier3: 基本面/因子数据（复用Project-Alpha因子库） =====
FACTOR_FIELDS: Final = [
    "turnover_rate", "pct_change", "pe_ttm", "pb_mrq", "ps_ttm",
    "pcf_ncf_ttm", "ep", "bp", "dp", "momentum_20", "reversal_5",
    "amplitude", "asset_turnover", "roe", "profit_growth", "revenue_growth", "roe_growth",
]

# 所有需要展开为30天滞后的特征字段（路线A不使用）
ALL_CROSS_SECTION_FIELDS: Final = PRICE_FIELDS + DERIVED_FIELDS + FACTOR_FIELDS

# ===== 序列参数 =====
SEQUENCE_LENGTH: Final = 30       # GRU的时序窗口长度（天）
N_FEATURES: Final = 6             # OHLC + VWAP + VOLUME（GRU序列输入）

# ===== 标签参数（路线B: 论文VWAP 10天设定） =====
# 论文定义: T+1到T+11的VWAP收益率（10天间隔）
# VWAP反映真实日内交易成本，更符合量价因子评估
LABEL_PERIOD: Final = 10          # T+1到T+11的收益率（论文: 10天）
LABEL_TYPE: Final = "vwap"        # "vwap"（量价路线B）或 "close"（路线A）

# ===== 预处理参数（论文表2） =====
MAD_MULTIPLIER: Final = 3         # 3倍MAD截断
FILLNA_VALUE: Final = 0           # 缺失值填充为0

# ===== MLP参数（路线B: 论文180维量价架构） =====
# 论文设定: MLP(180→128→128→128→1), BN+LeakyReLU, Dropout=0.1, LR=0.001, MSE Loss
# 180维需要正则化（Dropout/WD），与路线A小样本反直觉结论不同
MLP_LR: Final = 0.001              # 论文默认学习率（180维 vs 11维更需保守）
MLP_HIDDEN_DIMS: tuple = (128, 128, 128)  # 论文三级隐藏层架构
MLP_DROPOUT: Final = 0.1           # 180维需要Dropout防过拟合（论文默认）
MLP_MAX_EPOCHS: Final = 1000
MLP_PATIENCE: Final = 20           # 缩短早停（180维收敛更快）
MLP_BATCH_SIZE: Final = 1024       # 大batch训练（全量样本模式）

# ===== LightGBM参数（路线B: 论文表4参数，180维量价特征） =====
# 论文表4: learning_rate=0.1, max_depth=7, num_leaves=127, min_data=512
# 180维特征空间：深度7+叶127是论文调优结果，非路线A的小特征适配
LGBM_LR: Final = 0.1               # 论文表4学习率
LGBM_MAX_DEPTH: Final = 7          # 论文表4树深度（180维下可充分分裂）
LGBM_NUM_LEAVES: Final = 127       # 论文表4叶节点（2^7-1）
LGBM_MIN_DATA_IN_LEAF: Final = 512 # 论文表4最小样本（50只全A即约2560样本/截面）
LGBM_FEATURE_FRACTION: Final = 0.7 # 论文表4特征采样
LGBM_BAGGING_FRACTION: Final = 0.7 # 论文表4样本采样
LGBM_PATIENCE: Final = 50

# ===== GRU参数（Tier1优化: 增强正则化） =====
GRU_LR: Final = 0.001             # 可变学习率（CosineAnnealing或ReduceLROnPlateau）
GRU_HIDDEN_DIMS: Final = 64       # 论文2层GRU hidden=64
GRU_NUM_LAYERS: Final = 2
GRU_DROPOUT: Final = 0.3          # 增强Dropout对抗过拟合
GRU_MAX_EPOCHS: Final = 200
GRU_PATIENCE: Final = 10          # 缩短早停
GRU_BATCH_SIZE: Final = 1  # 截面DataLoader模式
GRU_LOSS_FN: Final = "mse"        # 论文设定: MSE（非IC）
GRU_LABEL_COL: Final = "label_vwap10"  # 论文设定: VWAP 10天标签

# ===== AGRU参数（Tier1优化: 增强正则化） =====
AGRU_LR: Final = 0.001
AGRU_HIDDEN_DIMS: Final = 64
AGRU_NUM_LAYERS: Final = 2
AGRU_DROPOUT: Final = 0.3          # 增强Dropout
AGRU_MAX_EPOCHS: Final = 200
AGRU_PATIENCE: Final = 10          # 缩短早停
AGRU_LOSS_FN: Final = "mse"        # 论文设定: MSE（非IC）
AGRU_LABEL_COL: Final = "label_vwap10"  # 论文设定: VWAP 10天标签

# ===== 集成参数 =====
ICIR_WINDOW: Final = 60           # ICIR加权窗口（60交易日）

# ===== 损失函数（路线B: 论文MSE设定） =====
# 论文统一使用MSE Loss（非IC Loss），量价特征不需要正交约束
LOSS_FN: Final = "mse"             # 论文设定: MSE（非路线A的IC Loss）
ORTH_LAMBDA: Final = 0.0           # 路线B: 无正交惩罚（论文不涉及）

# ===== 优化器正则化（路线B: 论文标准WD） =====
WEIGHT_DECAY: Final = 1e-5         # 论文默认弱正则化（180维适度防过拟合）

# ===== 训练/验证/测试时间划分 =====
TRAIN_END: Final = "2022-06-30"
VAL_END: Final = "2023-06-30"

# ===== 论文要求: 3个随机种子取平均 =====
RANDOM_SEEDS: Final = [42, 123, 456]

# ===== GPU =====
DEVICE: Final = "cuda" if _CUDA_AVAILABLE else "cpu"

# ===== 分组回测（论文用20组） =====
N_GROUPS: Final = 20

# ===== 防信息泄露：相邻数据集剔除天数 =====
BUFFER_DAYS: Final = 10           # 训练集/验证集/测试集相邻剔除10天

# ===== 滚动训练窗口（扩展窗口，论文图8） =====
# 数据实际从 2020-01-02 开始，因此窗口整体后移，避免训练集为空
# 每个元组: (train_end, val_end, test_end)
# 训练集: date < train_end - BUFFER_DAYS（扩展窗口）
# 验证集: train_end + BUFFER_DAYS <= date < val_end - BUFFER_DAYS（约半年）
# 测试集: val_end + BUFFER_DAYS <= date < test_end（约半年）
ROLLING_WINDOWS: Final = [
    ("2020-06-30", "2020-12-31", "2021-06-30"),  # W0
    ("2020-12-31", "2021-06-30", "2021-12-31"),  # W1
    ("2021-06-30", "2021-12-31", "2022-06-30"),  # W2
    ("2021-12-31", "2022-06-30", "2022-12-31"),  # W3
    ("2022-06-30", "2022-12-31", "2023-06-30"),  # W4
    ("2022-12-31", "2023-06-30", "2023-12-31"),  # W5
]

# ===== LightGBM特征名前缀（防特殊字符问题） =====
LGBM_FEATURE_PREFIX: Final = "f_"
