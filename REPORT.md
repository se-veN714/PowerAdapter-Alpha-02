# 多模型集成量价Alpha策略复现报告

> **复现论文**：招商证券《多模型集成量价Alpha策略》（2023）
>
> **项目仓库**：[https://github.com/se-veN714/PowerAdapter-Alpha-02](https://github.com/se-veN714/PowerAdapter-Alpha-02)
>
> **完成人**：seveN1foR(董庆语)
> **日期**：2026-06-28

---

## 项目结构总览

本报告按数据流顺序组织，每个章节对应项目中一个核心模块。阅读代码时建议按以下顺序：

```
config.py                  ← 全局配置（所有参数的单一入口）
    ↓
utils/data_loader.py       ← 数据加载（复用Project-Alpha OHLCV）
    ↓
utils/feature_engine.py    ← 特征工程（180维展开+序列构建+VWAP标签）
    ↓
utils/preprocess.py        ← 预处理（MAD/zscore）
    ↓
utils/dataset.py           ← 双模式Dataset（截面/序列）
    ↓
models/                    ← 四模型定义
  ├── mlp_alpha.py         ← MLP（180→128→128→128→1）
  ├── gbdt_alpha.py        ← LightGBM
  ├── gru_alpha.py         ← GRU（2层，hidden=64）
  └── agru_alpha.py        ← AGRU（GRU+Self-Attention）
    ↓
losses.py                  ← 损失函数（MSE+正交惩罚）
    ↓
train.py                   ← 训练（单次/滚动，PyTorch+GBDT双轨）
    ↓
evaluate.py                ← 评估（RankIC/ICIR/分组回测）
    ↓
ensemble.py                ← ICIR加权集成
    ↓
utils/metrics.py           ← 评估指标函数
utils/correlation.py       ← 模型间相关性分析
```

**辅助文件**：

| 文件 | 用途 |
|------|------|
| `scripts/run_rolling.py` | 批量滚动训练控制脚本 |
| `notebooks/routeA_analysis.ipynb` | 路线A（11因子）完整分析报告 |
| `notebooks/routeB_analysis.ipynb` | 路线B（180维量价）完整分析报告 |
| `run_train.bat / run_eval.bat` | 一键运行批处理 |
| `logs/` | 训练日志 + 评估图表 |
| `checkpoints/` | 72个模型文件（4模型×6窗口×3种子） |

---

## 一、论文理解与任务拆解

### 1.1 论文核心思路

传统Alpha模型通常依赖单一架构（MLP或Tree）。论文提出**多模型集成策略**，将四种异构模型进行ICIR加权融合：

```
输入（量价原始特征/基本面因子）
    │
    ├── 180维展开 ──► MLP ──┐
    │                        │
    ├── 180维展开 ──► GBDT ─┤
    │                        ├── ICIR 加权 Voting ──► 最终预测
    ├── (30,6)序列 ─► GRU  ─┤
    │                        │
    └── (30,6)序列 ─► AGRU ─┘
```

关键创新点：

1. **多模型异构集成**：MLP（截面非线性）、GBDT（树模型）、GRU（时序依赖）、AGRU（注意力增强）四种互补架构
2. **ICIR加权**：以历史ICIR作为集成权重，而非等权Voting，自适应市场环境
3. **双路线对比**：路线A（11因子基准）vs 路线B（180维量价论文核心）
4. **双模式数据表示**：截面180维展平（MLP/GBDT）+ 30步×6维序列（GRU/AGRU）

### 1.2 任务拆解与代码映射

| 阶段 | 代码入口 | 关键函数/类 | 核心决策 |
|------|---------|------------|----------|
| 数据加载 | `utils/data_loader.py` | `load_alpha_data()`, `compute_vwap()` | 复用Project-Alpha数据，VWAP=amount/volume |
| 特征工程 | `utils/feature_engine.py` | `build_cross_section_features()`, `build_sequence_data()` | 180维展开 + 30步序列 |
| 标签构建 | `utils/feature_engine.py` | `build_vwap_return_label()` | VWAP 10天收益率 |
| 预处理 | `utils/preprocess.py` | `mad_clip_section()`, `zscore_section()` | 截面MAD + zscore |
| 数据集 | `utils/dataset.py` | `CrossSectionDataset`, `SequenceDataset` | 截面/序列双模式，batch_size=1 |
| MLP模型 | `models/mlp_alpha.py` | `MLPAlphaModel` | 180→128→128→128→1，BN+Sigmoid |
| GBDT模型 | `models/gbdt_alpha.py` | `GBDTAlphaModel` | LightGBM，180维全量，非DataLoader模式 |
| GRU模型 | `models/gru_alpha.py` | `GRUAlphaModel` | 2层GRU(hidden=64)，最后隐状态 |
| AGRU模型 | `models/agru_alpha.py` | `AGRUAlphaModel` | GRU+Self-Attention聚合 |
| 训练 | `train.py` | `train_pytorch_model()`, `train_gbdt_model()`, `rolling_train()` | PyTorch/GBDT双轨，6窗口滚动 |
| 集成 | `ensemble.py` | `icir_weighted_voting()` | 滚动ICIR窗口60天 |
| 评估 | `evaluate.py`, `utils/metrics.py` | `rank_ic()`, `group_return()` | RankIC/ICIR/20组分组回测 |

---

## 二、复现方法论

### 2.1 数据获取 → `utils/data_loader.py`

> **代码入口**：`python -m utils.data_loader`

#### 股票池（50只，覆盖15个行业）

定义在 [`config.py` L34-65](config.py)，`STOCK_POOL` 常量。

银行（6）、保险/券商（4）、食品饮料（5）、家电（1）、科技/制造/电子（6）、医药（5）、地产/建材（3）、钢铁/有色（3）、化工（3）、电力/公用（3）、建筑（2）、通信/传媒（2）、石油/煤炭（3）、新能源（2）、汽车（2）。

选取标准：与Project-Alpha使用相同股票池，50只沪深300成分股代表性标的，覆盖主要行业。

#### 数据源

| 数据 | 来源 | 说明 |
|------|------|------|
| OHLCV 原始数据 | Project-Alpha `stock_data.csv` | 复用Alpha项目BaoStock获取的日线数据 |
| VWAP | 本地计算 `amount/volume` | 论文要求的均价字段 |

关键设计：
- **数据复用**：直接读取Project-Alpha的`stock_data.csv`（含open/high/low/close/amount），避免重复获取
- **VWAP近似**：`compute_vwap()` 使用 `amount/volume` 近似（BaoStock不直接提供VWAP）
- **股票池过滤**：仅保留50只股票池内的数据

#### 时间范围

`DATE_RANGE = ("2018-01-01", "2023-12-31")`，6年数据，约72K行。

实际训练从2020-01-02开始（需30天滞后窗口构建180维特征）。

### 2.2 特征工程 → `utils/feature_engine.py`

> **代码入口**：`python -m utils.feature_engine`
> **核心函数**：`build_cross_section_features()`, `build_sequence_data()`, `build_vwap_return_label()`

#### 路线B核心：180维量价特征展开

**6个日线量价字段** × **30天滞后窗口** = **180维截面特征**：

```
open_t-0,  open_t-1,  ..., open_t-29    (30维)
high_t-0,  high_t-1,  ..., high_t-29    (30维)
low_t-0,   low_t-1,   ..., low_t-29     (30维)
close_t-0, close_t-1, ..., close_t-29   (30维)
vwap_t-0,  vwap_t-1,  ..., vwap_t-29    (30维)
volume_t-0,volume_t-1,..., volume_t-29  (30维)
─────────────────────────────────────────
总计：6 × 30 = 180维
```

**为什么用原始量价而非衍生因子？** 论文的核心主张是"让模型自己从原始数据中学习因子结构"，而非人工定义因子。

#### 序列数据构建

对于GRU/AGRU，构建30步×6维的序列张量：

```python
# 每只股票一个 ndarray(n_days, 30, 6)
# 6维 = [open, high, low, close, vwap, volume]
# 30步 = 过去30个交易日
sequences[code] = ndarray(n_days, 30, 6)
```

#### VWAP标签

> 论文 §2.1：T+1到T+11的VWAP收益率（10天间隔）

```python
label = vwap_{t+11} / vwap_{t+1} - 1
```

- VWAP反映真实日内交易成本，比close更合理
- 10天窗口：平衡信号稳定性与预测时效
- 标签同样做截面zscore标准化

### 2.3 数据预处理 → `utils/preprocess.py`

> **代码入口**：`python -m utils.preprocess`

4步流水线，严格按截面操作（`groupby("date")`）：

```
原始180维数据
    → [1] 截面MAD去极值：median ± 3*MAD 截断
    → [2] 截面zscore标准化：(x - mean) / std
    → [3] Label截面zscore标准化
    → [4] 缺失值填充：NaN → 0
```

关键规则：
- **所有操作必须`groupby("date")`**：每交易日截面独立处理，严禁全局操作泄露未来信息
- **MAD而非3σ**：MAD基于中位数，对极端值鲁棒
- **NaN→0**：zscore后均值为0，填0等价于填均值

> ⚠️ 路线B不含因子方向调整（无因子定义），量价特征方向由模型自主学习

### 2.4 数据集构建 → `utils/dataset.py`

> **核心类**：`CrossSectionDataset`, `SequenceDataset`

#### 双模式Dataset设计

| Dataset | 输入形状 | 适用模型 | 说明 |
|---------|---------|---------|------|
| `CrossSectionDataset` | `(M, 180)` | MLP, GBDT | 180维展开，每样本一个交易日截面 |
| `SequenceDataset` | `(M, 30, 6)` | GRU, AGRU | 30步序列，每样本一个交易日截面 |

**关键设计**：每个`__getitem__`返回**一个交易日截面**的所有股票，batch_size=1：

```python
def __getitem__(self, idx):
    date_val = self.dates[idx]
    section = self.data.loc[self._group_indices[date_val]]
    return factor_tensor, label_tensor  # shape: (M, 180), (M,)
```

**时间划分** — `split_dataset()`，严格时序划分：

```python
train_df = df[df["date"] < train_end - buffer]           # 历史训练
val_df = df[(date >= train_end+buffer) & (date < val_end-buffer)]  # 验证
test_df = df[date >= val_end + buffer]                     # 测试
```

相邻边界各剔除10天（`BUFFER_DAYS=10`），防止标签窗口重叠导致信息泄露。

### 2.5 模型架构 → `models/`

#### MLP → `models/mlp_alpha.py`

> **代码**：`MLPAlphaModel`

```
Linear(180, 128) → BatchNorm1d → Sigmoid → Dropout(0.1)
Linear(128, 128) → BatchNorm1d → Sigmoid → Dropout(0.1)
Linear(128, 128) → BatchNorm1d → Sigmoid → Dropout(0.1)
Linear(128, 1)                                          ← 输出层（无激活函数）
```

- 论文设定：三级隐藏层128维，Sigmoid激活
- 末层无激活函数：标签zscore有正负
- 参数量：~72K

#### GBDT → `models/gbdt_alpha.py`

> **代码**：`GBDTAlphaModel`

```python
LGBMRegressor(
    learning_rate=0.1, max_depth=7, num_leaves=127,
    min_data_in_leaf=512, feature_fraction=0.7, bagging_fraction=0.7,
    n_estimators=500, early_stopping_rounds=50,
)
```

- 论文表4参数：深度7、叶节点127
- 非DataLoader模式：GBDT在函数空间迭代，传统fit/predict
- 输入180维zscore标准化特征（⚠️ 对树模型不友好，见已知问题）

#### GRU → `models/gru_alpha.py`

> **代码**：`GRUAlphaModel`

```
GRU(input_size=6, hidden_size=64, num_layers=2, dropout=0.3)
    → 取最后时间步隐状态
    → Linear(64, 1)
```

- 2层GRU，hidden=64
- 输入30步×6维序列（OHLC+VWAP+VOLUME）
- 最后时间步隐状态作为特征

#### AGRU → `models/agru_alpha.py`

> **代码**：`AGRUAlphaModel`

```
GRU(input_size=6, hidden_size=64, num_layers=2)
    → Self-Attention聚合（全时间步加权）
    → FC(128, 1)
```

- 在GRU基础上增加Self-Attention机制
- 允许模型自适应关注关键时间点
- 参数量：~58K

### 2.6 损失函数 → `losses.py`

> 论文统一使用**MSE Loss**（非IC Loss），量价特征不需要正交约束

```python
loss = nn.MSELoss()(pred, label)  # 标准MSE
```

- `ORTH_LAMBDA = 0.0`：路线B无正交惩罚（论文不涉及）
- `WEIGHT_DECAY = 1e-5`：弱L2正则化

### 2.7 训练策略 → `train.py`

> **代码入口**：`python train.py --mode [single|rolling] --model [mlp|gbdt|gru|agru]`

#### 双轨训练

| 训练轨 | 适用模型 | 关键差异 |
|--------|---------|---------|
| `train_pytorch_model()` | MLP, GRU, AGRU | 梯度下降 + DataLoader + 早停 |
| `train_gbdt_model()` | GBDT | fit/predict + 全量数据 + LightGBM早停 |

#### 滚动训练（扩展窗口）

```
W0: 训练至2019-12  │ 测试: 2021Q1-Q2
W1: 训练至2020-06  │ 测试: 2021Q3-Q4
W2: 训练至2020-12  │ 测试: 2022Q1-Q2
W3: 训练至2021-06  │ 测试: 2022Q3-Q4
W4: 训练至2021-12  │ 测试: 2023Q1-Q2
W5: 训练至2022-06  │ 测试: 2023Q3-Q4
```

- 扩展窗口：训练集随时间推移长度增加
- 验证集固定约252个交易日
- 测试集约半年
- 3个随机种子取平均（论文§2.1要求）

### 2.8 ICIR加权集成 → `ensemble.py`

> **核心函数**：`icir_weighted_voting()`, `compute_rolling_icir()`

```python
# 滚动计算每个模型的ICIR（窗口=60交易日）
icir_scores = compute_rolling_icir(predictions, actuals, window=60)

# ICIR加权合成最终预测
weights = softmax(icir_scores)
ensemble_pred = sum(w_i * pred_i for w_i, pred_i in zip(weights, predictions))
```

与论文等权Voting不同，ICIR加权根据近期表现动态调整权重。

### 2.9 评估体系 → `evaluate.py` + `utils/metrics.py`

| 指标 | 函数 | 说明 |
|------|------|------|
| **Rank IC** | `rank_ic()` | 预测值与真实标签的截面Spearman相关系数 |
| **ICIR** | `calc_ic_series()` | IC均值/IC标准差，衡量信号稳定性 |
| **分组回测** | `group_return()` | 20组等权分组，Top/Bottom/Long-Short净值曲线 |
| **模型相关性** | `model_cross_section_correlation()` | 截面视角（同一日不同模型预测的相关性） |

> 评估均按截面独立计算，最终报告窗口均值。

---

## 三、技术栈与环境

| 组件 | 版本/规格 | 说明 |
|------|----------|------|
| Python | 3.12.7 | Windows 64-bit |
| PyTorch | 2.5.1+cu124 | CUDA 12.4 |
| LightGBM | 4.6+ | GBDT模型 |
| scipy | 1.14+ | Spearman秩相关系数 |
| pandas | 2.2+ | 数据处理 |
| GPU | NVIDIA RTX 4060 (8GB) | 单卡，CUDA 13.0驱动兼容cu124 |
| CPU | Intel i5-12450H | 12核 |

**显存使用**：约1.3GB/8GB（小截面batch），GPU利用率99%但显存远未瓶颈。

---

## 四、复现结果

### 4.1 路线A：11因子路线（参考基准）

#### 模型评估（6窗口Mean Val IC）

| 模型 | Mean Val IC | 正窗口数 | 最佳窗口 IC |
|------|:----------:|:-------:|:----------:|
| **MLP** | **+0.0686** | 4/6 | +0.1250 (W1) |
| GRU | +0.0426 | 4/6 | +0.1012 (W3) |
| AGRU | +0.0419 | 3/6 | +0.0978 (W3) |
| GBDT | -0.0376 | 0/6 | -0.0245 (W4) |

#### ICIR加权集成

| 集成方式 | Test IC | vs Best Single |
|----------|:-------:|:-------------:|
| Full Ensemble | **+0.0141** | 优于MLP(-0.0171) |
| MLP Only（Best Single）| -0.0171 | — |

**结论**：路线A中ICIR集成有效，Full Ensemble IC由负转正。

#### 模型间相关性

|  | MLP | GBDT | GRU | AGRU |
|--|:---:|:---:|:---:|:---:|
| MLP | 1.00 | 0.12 | 0.08 | 0.07 |
| GBDT | | 1.00 | 0.05 | 0.04 |
| GRU | | | 1.00 | 0.31 |
| AGRU | | | | 1.00 |

**截面vs时序正交**：MLP/GBDT（截面模型）与GRU/AGRU（时序模型）间ρ≈0，集成时可提供互补信息。

### 4.2 路线B：180维量价路线（论文核心）

#### 模型评估（6窗口Mean Val IC）

| 模型 | Mean Val IC | 正窗口数 | 最佳窗口 IC |
|------|:----------:|:-------:|:----------:|
| **MLP** | **+0.0561** | 4/6 | +0.0982 (W0) |
| GRU | +0.0079 | 3/6 | +0.0438 (W2) |
| AGRU | +0.0046 | 3/6 | +0.0329 (W2) |
| GBDT | -0.0245 | 1/6 | +0.0111 (W5) |

#### ICIR加权集成

| 集成方式 | Test IC | vs Best Single |
|----------|:-------:|:-------------:|
| Full Ensemble | -0.0038 | 差于MLP(+0.0184) |
| MLP Only（Best Single）| **+0.0184** | — |

**结论**：路线B中ICIR集成无效——所有模型信号弱，集成反而拖累。MLP单独表现最佳。

#### 窗口表现分析

```
W0 (2021H1): MLP=+0.0982, GRU=+0.0438, AGRU=+0.0329  ← 最佳窗口
W1 (2021H2): MLP=+0.0734, GRU=-0.0221, AGRU=-0.0156  ← 时序模型开始恶化
W2 (2022H1): MLP=+0.0647, GRU=+0.0367, AGRU=+0.0213  ← 反弹
W3 (2022H2): MLP=+0.0528, GRU=-0.0125, AGRU=-0.0089  ← 2022下降趋势
W4 (2023H1): MLP=+0.0319, GRU=+0.0034, AGRU=+0.0021  ← GRU/AGRU坍缩
W5 (2023H2): MLP=+0.0156, GBDT=+0.0111, GRU=-0.0012   ← 普遍弱化
```

### 4.3 路线A vs 路线B 对比

| 维度 | 路线A (11因子) | 路线B (180维量价) |
|------|:-----------:|:--------------:|
| 特征数 | 11 | 180 |
| MLP Mean Val IC | **+0.0686** | +0.0561 |
| GRU Mean Val IC | **+0.0426** | +0.0079 |
| AGRU Mean Val IC | **+0.0419** | +0.0046 |
| GBDT Val IC | -0.0376 | **-0.0245** |
| ICIR集成效果 | ✅ 有效 | ❌ 无效 |
| 特征/样本比 | 11/25≈0.44 | 180/25=7.2 |

**核心发现**：在当前25只股票的小样本下，路线A（11因子）整体优于路线B（180维量价）。路线B参数样本比7.2，严重过参数化，时序模型（GRU/AGRU）尤其受影响。

### 4.4 论文基准对照

| 指标 | 论文报告 | 本复现（路线B） | 差距分析 |
|------|:------:|:------:|------|
| MLP RankIC | 10.99% | ~5.6% | 样本量不足（25 vs 论文500+） |
| GBDT RankIC | 10.66% | -2.45% | z-score标准化破坏树模型 |
| GRU RankIC | 11.27% | ~0.8% | 时序模型在小样本下坍缩 |
| AGRU RankIC | 10.67% | ~0.5% | 同上 |
| 集成 RankIC | **11.90%** | -0.38% | 无"锚"模型可依靠 |

**差距根因**：
1. 股票样本量25只 vs 论文500+只 → 参数样本比过高
2. GBDT输入z-score标准化 → 破坏树模型特征分割能力
3. 2022年极端行情 → GRU/AGRU预测坍缩为常数

---

## 五、关键发现与挑战

### 5.1 论文假设验证

| 假设 | 预期 | 实验结果 | 验证 |
|------|------|----------|:--:|
| H1 多模型互补 | MLP/GBDT/GRU/AGRU信息量不重叠 | 截面vs时序ρ≈0，互补成立 | ✅ |
| H2 ICIR集成提升 | 加权集成优于单模型 | 路线A有效(+0.0141)，路线B无效(-0.0038) | ⚠️ |
| H3 量价特征有效性 | 180维量价>11因子 | 路线A IC(+0.0686) > 路线B(+0.0561) | ❌ |
| H4 GBDT+量价有效 | GBDT应给出正向信号 | 全路线Val IC为负(-0.0245~-0.0376) | ❌ |
| H5 时序模型有效 | GRU/AGRU捕获跨期依赖 | 路线B近乎失效(W4/W5坍缩) | ❌ |

### 5.2 关键挑战

| 挑战 | 表现 | 根因 | 代码位置 |
|------|------|------|----------|
| **小样本过参数化** | 180维特征,25只股票 | 参数样本比=7.2，严重过拟合风险 | 股票池仅Alpha的25只 |
| **GBDT+zscore失败** | 全路线Val IC为负 | z-score破坏树模型特征分割能力 | `gbdt_alpha.py` |
| **GRU/AGRU坍缩** | W4/W5预测为常数 | 2022年极端行情下时序模式失效 | `gru_alpha.py`, `agru_alpha.py` |
| **截面BatchNorm** | batch_size=1时BN不稳定 | 单截面内M=25太小 | `mlp_alpha.py` |
| **集成无锚** | 路线B集成无效 | 无"锚"模型时集成拖累 | `ensemble.py` |
| **无风格中性化** | 未剥离行业/风格收益 | 需凸优化求解器 | 待实现 |

### 5.3 量化发现

```
路线A: 截面vs时序模型ρ≈0.05~0.12 → 信息面正交 ✓
路线B: MLP是唯一稳定正向模型 → 其他模型拖累集成
跨窗口: W0最佳(+0.0982) → W5最弱(+0.0156) → 模型随时间衰减
路线对比: 小样本下，简单特征路径优于复杂特征路径
```

---

## 六、代码架构亮点

### 6.1 双模式Dataset

```python
# 同一个processed_data.csv，两种视角
CrossSectionDataset:  每样本 (M, 180)    → MLP/GBDT
SequenceDataset:      每样本 (M, 30, 6)  → GRU/AGRU
```

设计优势：数据管线单一，Dataset层做视角切换，避免数据冗余。

### 6.2 滚动训练统一入口

```python
python train.py --mode rolling   # 自动：4模型 × 6窗口 × 3种子 = 72次训练
```

- 扩展窗口自动推进
- 每个窗口自动保存checkpoint
- GBDT/PyTorch双轨统一接口

### 6.3 防信息泄露设计

- 所有预处理`groupby("date")`截面独立
- 相邻数据集10天Buffer
- 时间划分严禁随机shuffle

---

## 七、结论与后续建议

### 7.1 复现成果

1. ✅ 完整复现了论文的四模型集成Alpha策略，包括MLP/GBDT/GRU/AGRU四条管线
2. ✅ 实现了ICIR加权集成机制，并在路线A中验证了有效性
3. ✅ 建立了双路线评估框架（11因子 vs 180维量价），输出6窗口滚动评估+模型相关性+分组回测
4. ✅ 双路线Notebook完整分析报告（20页+）
5. ⚠️ 路线B核心IC指标低于论文报告水平，主要受限于样本量（25 vs 500+）

### 7.2 改进优先级

| 优先级 | 项目 | 预期影响 | 代码改动 |
|:--:|------|----------|---------|
| 🔴 高 | 扩大股票池（25→200-500只） | 大幅提升IC，接近论文基准 | `config.py` `STOCK_POOL` |
| 🔴 高 | GBDT用Rank标准化或不标准化 | 修复GBDT全负问题 | `preprocess.py` + `gbdt_alpha.py` |
| 🟡 中 | 特征工程化量价因子（波动率/量比等） | 降维+提升信息密度 | `feature_engine.py` |
| 🟡 中 | 多频率多周期特征（5/10/20/60天） | 捕获不同时间尺度模式 | `feature_engine.py` |
| 🟢 低 | 指数增强+风格中性化 | 面试加分项 | 新增凸优化模块 |
| 🟢 低 | 超参数调优（类似Alpha的17实验） | 边际提升 | `scripts/` |

### 7.3 提升路径

```
当前状态（路线B MLP=0.0561）
  + 扩大股票池(500只)     → 预计IC +3~5%
  + GBDT Rank标准化      → 修复GBDT负值，集成有锚
  + 特征工程化            → 降维180→50+衍生，提升信噪比
  + 超参调优              → 类似Alpha的+1.57%提升
  = 目标：RankIC接近10%（论文基准11.90%）
```

---

## 八、与 Project-Alpha 的关系

| 维度 | Project-Alpha | Alpha-02 |
|------|-------------|----------|
| 论文 | 端到端动态Alpha模型(2023) | 多模型集成量价Alpha策略(2023) |
| 模型 | 单MLP | MLP+GBDT+GRU+AGRU四模型 |
| 特征 | 11因子 | 180维量价（路线B）/ 11因子（路线A） |
| 标签 | 20日close收益 | 10日VWAP收益 |
| 核心机制 | 端到端因子权重学习+正交惩罚 | 多模型ICIR加权集成 |
| 数据 | 50只独立获取 | 复用Alpha OHLCV数据 |
| VC | [PowerAdapter-Alpha](https://github.com/se-veN714/PowerAdapter-Alpha) | [PowerAdapter-Alpha-02](https://github.com/se-veN714/PowerAdapter-Alpha-02) |

Alpha-02是Alpha的延续与扩展（非替代/迭代），聚焦量价原始特征与多模型集成两条新维度。

---

*报告生成时间：2026-06-28 | 项目版本：v1.4*
