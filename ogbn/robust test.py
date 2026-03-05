# robustness_comparison - Academic Version (Top Label & Bottom Sub-caption)
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 0. 学术绘图全局配置
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.2,
    "lines.markersize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ==========================================
# 1. 实验数据配置
# ==========================================
noise_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

arxiv_data = {
    'GCN': [0.7027, 0.682, 0.615, 0.531, 0.435, 0.352, 0.284],
    'GAT': [0.700, 0.678, 0.648, 0.584, 0.502, 0.421, 0.357],
    'PANS': [0.7484, 0.749, 0.743, 0.711, 0.684, 0.639, 0.548]
}

products_data = {
    'GCN': [0.7352, 0.7110, 0.690, 0.674, 0.577, 0.382, 0.324],
    'GAT': [0.7210, 0.7118, 0.688, 0.644, 0.552, 0.461, 0.387],
    'PANS': [0.7912, 0.770, 0.771, 0.731, 0.724, 0.706, 0.601]
}

# ==========================================
# 2. 绘图逻辑
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

def apply_academic_style(ax):
    """应用学术风格：标签置顶，边框浅灰"""
    grey_color = '#D3D3D3'
    for spine in ax.spines.values():
        spine.set_color(grey_color)
        spine.set_linewidth(1.2)
    ax.grid(False)
    
    # 【修改点 1】：将噪声比标签移至上方
    ax.set_xlabel('Structural Noise Ratio ($\eta$)', labelpad=10)
    ax.xaxis.set_label_position('top') 
    
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0.2, 0.85)

# 子图 1: ogbn-arxiv
ax1.plot(noise_ratios, arxiv_data['GCN'], marker='s', color='#d62728', linestyle='--', label='GCN Baseline')
ax1.plot(noise_ratios, arxiv_data['GAT'], marker='^', color='#2ca02c', linestyle='-.', label='GAT Baseline')
ax1.plot(noise_ratios, arxiv_data['PANS'], marker='o', color='#1f77b4', linestyle='-',  label='Our Method (PANS)')
apply_academic_style(ax1)

# 子图 2: ogbn-products
ax2.plot(noise_ratios, products_data['GCN'], marker='s', color='#d62728', linestyle='--', label='GCN Baseline')
ax2.plot(noise_ratios, products_data['GAT'], marker='^', color='#2ca02c', linestyle='-.', label='GAT Baseline')
ax2.plot(noise_ratios, products_data['PANS'], marker='o', color='#1f77b4', linestyle='-',  label='Our Method (PANS)')
apply_academic_style(ax2)

# 【修改点 2】：手动添加子图标注 (a) 和 (b) 在下方，ha='center' 居中
ax1.text(0.5, -0.18, '(a) ogbn-arxiv', transform=ax1.transAxes, 
         fontsize=16, ha='center', va='top')
ax2.text(0.5, -0.18, '(b) ogbn-products', transform=ax2.transAxes, 
         fontsize=16, ha='center', va='top')

# 统一底部图例
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))

# 【修改点 3】：调整布局布局，rect 参数确保顶部标签不被遮挡
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

# 保存 PDF
plt.savefig('robust.pdf')
print("学术级对比图已更新生成：robust.pdf")