# -*- coding: utf-8 -*-
"""P3: Generate all figures for reproduction notebook."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
OUTPUT_DIR = LOG_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rolling = json.load(open(LOG_DIR / 'rolling_results_20260626_020913.json'))
ensemble = json.load(open(LOG_DIR / 'ensemble_results_20260626_022609.json'))
correlation = json.load(open(LOG_DIR / 'correlation_20260626_023719.json'))
eval_rolling = json.load(open(LOG_DIR / 'eval_rolling_results.json'))

MODEL_ORDER = ['mlp', 'gbdt', 'gru', 'agru']
MODEL_LABELS = ['MLP', 'GBDT', 'GRU', 'AGRU']

# ========== Fig 1: Rolling Val IC ==========
val_ic_data = {}
for m in MODEL_ORDER:
    val_ic_data[m] = [np.mean([s['best_val_ic'] for s in rolling[m][w]['seeds']]) for w in range(6)]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(6)
width = 0.18
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
for i, (m, label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
    ax.bar(x + i * width, val_ic_data[m], width, label=label, color=colors[i], edgecolor='white', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Rolling Window'); ax.set_ylabel('Val RankIC')
ax.set_title('4-Model Rolling Validation RankIC (3-Seed Average)')
ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rolling_val_ic.png'); plt.close()
print('[OK] rolling_val_ic.png')

# ========== Fig 2: Ensemble Comparison ==========
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['MLP Single', 'CS (MLP+GBDT)', 'SEQ (GRU+AGRU)', 'Full (4-Model)']
values = [-0.0171, 0.0028, -0.0003, 0.0141]
bar_colors = ['#95A5A6', '#2E86AB', '#F18F01', '#27AE60']
ax.bar(labels, values, color=bar_colors, edgecolor='white', linewidth=1, width=0.55)
ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
ax.set_ylabel('Mean Test IC (18 tasks)')
ax.set_title('ICIR-Weighted Ensemble: Test Set Mean IC')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ensemble_comparison.png'); plt.close()
print('[OK] ensemble_comparison.png')

# ========== Fig 3: Correlation Analysis ==========
corr_matrix = correlation['mean_matrix']
models_corr = ['mlp', 'gbdt', 'gru', 'agru']
labels_corr = ['MLP', 'GBDT', 'GRU', 'AGRU']
corr_arr = np.zeros((4, 4))
for i, mi in enumerate(models_corr):
    for j, mj in enumerate(models_corr):
        corr_arr[i, j] = corr_matrix[mi][mj]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cmap = LinearSegmentedColormap.from_list('custom_corr', ['#2166AC', '#F7F7F7', '#B2182B'], N=256)
im = axes[0].imshow(corr_arr, cmap=cmap, vmin=-0.1, vmax=1.0, aspect='equal')
axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
axes[0].set_xticklabels(labels_corr); axes[0].set_yticklabels(labels_corr)
axes[0].set_title('Global Mean Correlation Matrix\n(18 Tasks, Test Set)')
for i in range(4):
    for j in range(4):
        color = 'white' if abs(corr_arr[i, j]) > 0.6 else 'black'
        axes[0].text(j, i, f'{corr_arr[i, j]:.3f}', ha='center', va='center', fontsize=12, fontweight='bold', color=color)
plt.colorbar(im, ax=axes[0], shrink=0.8)

windows = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5']
wlabels_short = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5']
mlp_gbdt = [correlation['per_window'][w]['mean']['mlp']['gbdt'] for w in windows]
gru_agru = [correlation['per_window'][w]['mean']['gru']['agru'] for w in windows]
cs_vs_seq = []
for w in windows:
    mat = correlation['per_window'][w]['mean']
    cs_vs_seq.append(np.mean([mat['mlp']['gru'], mat['mlp']['agru'], mat['gbdt']['gru'], mat['gbdt']['agru']]))
axes[1].plot(wlabels_short, mlp_gbdt, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='MLP vs GBDT (intra-CS)')
axes[1].plot(wlabels_short, gru_agru, 's-', color='#A23B72', linewidth=2, markersize=8, label='GRU vs AGRU (intra-SEQ)')
axes[1].plot(wlabels_short, cs_vs_seq, 'D--', color='#27AE60', linewidth=2, markersize=8, label='CS vs SEQ (cross-type)')
axes[1].axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
axes[1].set_xlabel('Rolling Window'); axes[1].set_title('Model Correlation by Window')
axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3); axes[1].set_ylim(-0.15, 0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_analysis.png'); plt.close()
print('[OK] correlation_analysis.png')

# ========== Fig 4: 20-Group Backtest (4 models) ==========
model_groups = {}
for m in MODEL_ORDER:
    groups = eval_rolling[m]['group_return']
    numeric_groups = sorted([g for g in groups if g['group'] != 'long_short'], key=lambda x: int(x['group']))
    model_groups[m] = numeric_groups

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, (m, label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
    ax = axes[idx // 2, idx % 2]
    returns = [g['mean_return'] for g in model_groups[m]]
    bar_colors = ['#C0392B' if r < 0 else '#27AE60' for r in returns]
    bar_colors[0] = '#8B0000'; bar_colors[-1] = '#006400'
    ax.bar(range(1, 21), returns, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.set_title(f'{label} 20-Group Backtest')
    ax.set_xlabel('Group (1=Short, 20=Long)'); ax.set_ylabel('Mean Return')
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('20-Group Backtest by Model (All 6 Windows x 3 Seeds Aggregated)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'group_backtest_models.png'); plt.close()
print('[OK] group_backtest_models.png')

# ========== Fig 5: Full Ensemble 20-Group ==========
full_groups = {i: [] for i in range(1, 21)}
for task_name, task_data in ensemble['per_task'].items():
    if 'full_ensemble' not in task_data:
        continue
    for g in task_data['full_ensemble']['group_return']:
        if g['group'] != 'long_short':
            full_groups[g['group']].append(g['mean_return'])
full_avg = {k: np.mean(v) for k, v in full_groups.items()}
full_std = {k: np.std(v) for k, v in full_groups.items()}

fig, ax = plt.subplots(figsize=(12, 7))
returns_list = [full_avg[g] for g in range(1, 21)]
stds_list = [full_std[g] for g in range(1, 21)]
colors_ens = ['#C0392B' if r < 0 else '#27AE60' for r in returns_list]
colors_ens[0] = '#8B0000'; colors_ens[-1] = '#006400'
ax.bar([str(i) for i in range(1, 21)], returns_list, color=colors_ens, edgecolor='white',
       linewidth=0.8, yerr=stds_list, capsize=3)
ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.5)
ax.set_xlabel('Group (1=Short, 20=Long)'); ax.set_ylabel('Mean Return (+-1std)')
ax.set_title('Full Ensemble 20-Group Backtest (18 Tasks Aggregated)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'group_backtest_ensemble.png'); plt.close()
print('[OK] group_backtest_ensemble.png')

print(f'\nAll 5 figures saved to {OUTPUT_DIR.absolute()}')
