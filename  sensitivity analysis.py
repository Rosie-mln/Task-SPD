# sensitivity_analysis - Academic Version (Top Label & Bottom Sub-caption)
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 0. 学术绘图全局配置
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,               # 针对 1x3 布局优化字号
    "axes.labelsize": 20,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "lines.linewidth": 2.5,
    "lines.markersize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ==========================================
# 1. 实验数据
# ==========================================
rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
arxiv_acc = [0.734, 0.732, 0.749, 0.741, 0.735, 0.738, 0.721, 0.721, 0.720]
product_acc = [0.765, 0.776, 0.795, 0.793, 0.781, 0.776, 0.780, 0.771, 0.775]
yelp_f1 = [0.775, 0.810, 0.790, 0.782, 0.778, 0.752, 0.750, 0.747, 0.731]

# ==========================================
# 2. 绘图核心逻辑
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

def decorate_ax(ax, ylabel, ylim):
    """应用学术风格：标签置顶，边框浅灰"""
    grey_color = '#D3D3D3'
    for spine in ax.spines.values():
        spine.set_color(grey_color)
        spine.set_linewidth(1.2)
    ax.grid(False)
    
    # 将稀疏率标签移至上方
    ax.set_xlabel(r'Sparsity Ratio ($\rho$)', labelpad=12)
    ax.xaxis.set_label_position('top') 
    
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.legend(loc='lower right', frameon=False)

# --- 图 (a) ogbn-arxiv ---
axes[0].plot(rho_values, arxiv_acc, marker='o', color='#1f77b4', linestyle='-', label='ours', zorder=3)
decorate_ax(axes[0], 'Accuracy', (0.65, 0.80))
axes[0].text(0.5, -0.22, '(a) ogbn-arxiv', transform=axes[0].transAxes, fontsize=20, ha='center', va='top')

# --- 图 (b) ogbn-products ---
axes[1].plot(rho_values, product_acc, marker='s', color='#d62728', linestyle='-', label='ours', zorder=3)
decorate_ax(axes[1], 'Accuracy', (0.74, 0.82))
axes[1].text(0.5, -0.22, '(b) ogbn-products', transform=axes[1].transAxes, fontsize=20, ha='center', va='top')

# --- 图 (c) YelpChi ---
axes[2].plot(rho_values, yelp_f1, marker='D', color='#2ca02c', linestyle='-', label='ours', zorder=3)
decorate_ax(axes[2], 'Macro-F1 Score', (0.70, 0.85))
axes[2].text(0.5, -0.22, '(c) YelpChi', transform=axes[2].transAxes, fontsize=20, ha='center', va='top')

# 调整布局以适应顶部标签和底部文字
plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# 保存 PDF
plt.savefig('sensitivity_analysis.pdf')
print("学术级敏感性分析图表已生成：sensitivity_analysis.pdf")