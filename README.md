# Alpha-02

复现招商证券《多模型集成量价Alpha策略》(2023)，MLP/GBDT/GRU/AGRU 四模型 ICIR 加权集成。

> **项目定位**: Alpha-02 是 [Project-Alpha](https://github.com/se-veN714/PowerAdapter-Alpha) 的延续与扩展，聚焦量价原始特征路线（非替代/迭代）。

## 环境要求

- Python 3.12.7
- NVIDIA RTX 4060 (8GB) + CUDA 12.4+
- LightGBM 4.6+
- Windows 64-bit

## 快速开始

### 1. 创建虚拟环境

```powershell
cd D:\.shigodo\shigodo\Quantification\Project-Alpha-02
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装 PyTorch（CUDA 版）

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

验证 GPU：

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 4060
```

### 3. 安装其余依赖

```powershell
pip install -r requirements.txt
```

### 4. 数据管线

> 复用 Project-Alpha 的 OHLCV 原始数据，VWAP = amount/volume 近似计算。

```powershell
python -m utils.data_loader       # 加载数据 + 计算VWAP
python -m utils.feature_engine    # 特征工程：180维展开 + 30步序列构建
python -m utils.preprocess        # MAD去极值 + 截面zscore标准化
```

### 5. 运行流程

```powershell
# 单模型训练
python train.py --mode single --model mlp
python train.py --mode single --model gbdt
python train.py --mode single --model gru
python train.py --mode single --model agru

# 滚动训练（4模型 × 6窗口 × 3种子 = 72次训练）
python train.py --mode rolling

# 单模型评估
python evaluate.py --model mlp

# ICIR 加权集成评估
python ensemble.py

# 模型间相关性分析
python run_correlation.py
```

### 6. 一键运行

```powershell
run_train.bat        # 全量滚动训练
run_eval.bat         # 全量评估
run_ensemble.bat     # ICIR 集成
run_correlation.py   # 相关性分析
```

## 核心思路

### 路线 A：11因子路线（参考基准）
11个基本面+技术因子 → MLP/GBDT/GRU/AGRU → ICIR 加权集成

### 路线 B：180维量价路线（论文核心）
6个日线量价字段（OHLC + VWAP + VOLUME）× 30天滞后展开 → 180维截面特征

```
raw OHLCV data
     │
     ▼
┌──────────────────────────────────────┐
│  feature_engine.py                   │
│  每个交易日，每只股票：                │
│  - 过去30天 open/high/low/close/     │
│    vwap/volume 展开 → 180维          │
│  - 过去30天 6字段序列 → (30,6)       │
│  - 标签: T+1到T+11 VWAP 收益率       │
└──────────────────────────────────────┘
     │
     ├─── 180维 ──► MLP / GBDT
     │
     └─── (30,6) ─► GRU / AGRU
              │
              ▼
     ┌──────────────────┐
     │  ICIR 加权集成    │
     │  Ensemble Vote    │
     └──────────────────┘
```

## 模型架构

| 模型 | 输入 | 架构 | 特点 |
|------|------|------|------|
| **MLP** | (N, 180) | 180→128→128→128→1 | BN + LeakyReLU, No Dropout |
| **GBDT** | (N, 180) | LightGBM, n_estimators=200 | 树模型，原始尺度更好 |
| **GRU** | (N, 30, 6) | 2层 GRU, hidden=64 | 时序建模，跨期依赖 |
| **AGRU** | (N, 30, 6) | GRU + Self-Attention | 注意力增强，关键时间点 |

## 论文基准指标

| 指标 | MLP | GBDT | GRU | AGRU | **集成** |
|------|-----|------|-----|------|----------|
| RankIC | 10.99% | 10.66% | 11.27% | 10.67% | **11.90%** |
| ICIR | 1.17 | 1.14 | 1.12 | 1.01 | **1.13** |
| 多头收益 | 33.23% | 29.84% | 31.28% | 24.53% | **33.11%** |

## 项目结构

```
Project-Alpha-02/
├── README.md                # 本文档
├── config.py                # 全局参数配置
├── train.py                 # 训练入口（single/rolling）
├── evaluate.py              # 评估入口（RankIC/ICIR/分组回测）
├── ensemble.py              # ICIR 加权集成
├── losses.py                # 损失函数（MSE + 正交惩罚）
├── requirements.txt         # Python 依赖
├── setup_env.bat            # 环境初始化脚本
├── utils/
│   ├── data_loader.py       # 数据加载 + VWAP计算
│   ├── feature_engine.py    # 180维展开 + 序列构建 + 标签
│   ├── preprocess.py        # MAD去极值 + zscore标准化
│   ├── dataset.py           # CrossSectionDataset + SequenceDataset
│   ├── metrics.py           # RankIC/ICIR/分组回测
│   └── correlation.py       # 模型间相关性
├── models/
│   ├── mlp_alpha.py         # MLP（全连接）
│   ├── gbdt_alpha.py        # LightGBM
│   ├── gru_alpha.py         # GRU（2层）
│   └── agru_alpha.py        # AGRU（GRU+Attention）
├── scripts/
│   ├── init_project.py      # 项目初始化
│   └── run_rolling.py       # 批量滚动训练控制
├── notebooks/
│   ├── routeA_analysis.ipynb  # 路线A 分析
│   └── routeB_analysis.ipynb  # 路线B 分析
├── data/                    # 数据目录
│   ├── raw/                 # 原始 OHLCV 数据
│   └── processed/           # 预处理后数据
├── checkpoints/             # 模型保存
├── logs/                    # 训练日志 + 评估图表
└── run_*.bat                # 一键运行脚本
```

## 与 Project-Alpha 的关系

| Project-Alpha | Alpha-02 |
|---------------|----------|
| 11个基本面+技术因子 | 180维量价原始特征（路线B）/ 11因子（路线A） |
| 单 MLP 模型 | MLP + GBDT + GRU + AGRU 四模型集成 |
| 端到端因子权重学习 | 多模型 ICIR 加权 Voting |
| 1天标签 | 10天 VWAP 标签 |
| `线性基准` + `正交惩罚` | 仅保留正交惩罚 |

Alpha-02 复用 Project-Alpha 的 OHLCV 原始数据（50只股票，2018-2023），VWAP 由 amount/volume 近似计算。

## License

MIT
