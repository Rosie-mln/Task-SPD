# 标签重用

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import negative_sampling, softmax 
from torch_geometric.loader import NeighborLoader 
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# global参数配置
# ==========================================
class Config:
    hidden_dim = 128     
    proj_dim = 256       
    hops = 2             
    epochs = 100         # 建议先跑100轮观察
    soft_end = 40        
    lr_decay_epoch = 60 
    lr_init = 0.001      
    batch_size = 512    
    target_sparsity = 0.40 
    lambda_max = 2.0       
    link_weight = 0.7      
    weight_decay = 1e-4    

# --- SubgraphEnhancer (Local-Softmax) ---
class SubgraphEnhancer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, weights):
        x = self.lin(x)
        norm_weights = softmax(weights, edge_index[1], num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, weights=norm_weights)

    def message(self, x_j, weights):
        return weights.view(-1, 1) * x_j

class BilinearSampler(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.W = nn.Parameter(torch.Tensor(proj_dim, proj_dim))
        nn.init.xavier_uniform_(self.W)
    def forward(self, h_i, h_j):
        z_i = self.proj(h_i)
        z_j = self.proj(h_j)
        score = torch.sum((z_i @ self.W) * z_j, dim=-1) 
        return torch.sigmoid(score)

# ==========================================
# 修正后的 NeuralRecursiveSystem
# ==========================================
class NeuralRecursiveSystem(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, hops=2, tau=0.8): 
        super().__init__()
        # 保持 hidden_size * 8 = 1024
        self.total_hidden = hidden_size * 8
        
        # 标签嵌入层
        self.label_emb = nn.Embedding(out_size + 1, self.total_hidden)
        
        # 语义编码器
        self.gat1 = GATConv(in_size, hidden_size, heads=8, dropout=0.6)
        self.ln1 = nn.LayerNorm(self.total_hidden)
        
        self.gat2 = GATConv(self.total_hidden, hidden_size, heads=8, dropout=0.6)
        self.ln2 = nn.LayerNorm(self.total_hidden)
        
        # 采样器与增强器
        self.sampler_net = BilinearSampler(in_dim=self.total_hidden, proj_dim=Config.proj_dim)
        self.enhancer = SubgraphEnhancer(self.total_hidden, self.total_hidden)
        
        # 【关键修复】：残差投影层，确保所有项都能对齐到 total_hidden
        self.res_lin = nn.Linear(in_size, self.total_hidden)
        
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.total_hidden * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.total_hidden, hidden_size * 4),
            torch.nn.BatchNorm1d(hidden_size * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_size * 4, out_size)
        )
        self.hops = hops
        self.tau = tau

    def get_neural_recursive_weights(self, h, edge_index, start_mask, hard=True):
        row, col = edge_index
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)
        scores = self.sampler_net(h[row], h[col])
        final_weights = torch.zeros(num_edges, device=edge_index.device)
        active_nodes = start_mask.float() 
        for h_step in range(self.hops):
            active_edges_mask = active_nodes[row]
            sampling_logits = torch.stack([1 - scores, scores], dim=-1).clamp(min=1e-9).log()
            sampling_mask = F.gumbel_softmax(sampling_logits, tau=self.tau, hard=hard, dim=-1)[:, 1]
            current_step_weights = sampling_mask * active_edges_mask
            final_weights = torch.max(final_weights, current_step_weights)
            new_active_nodes = torch.zeros(num_nodes, device=edge_index.device)
            new_active_nodes.scatter_add_(0, col, current_step_weights)
            active_nodes = (new_active_nodes > 1e-5).float() 
        return final_weights

    def forward(self, x, edge_index, y_label, target_mask, hard=True):

        # 只有对齐后的 x_proj (1024维) 才能和 y_emb (1024维) 相加
        x_proj = self.res_lin(x) 
        y_emb = self.label_emb(y_label)
        
        # 融合：原始投影 + 标签先验
        x_fused = x_proj + y_emb 
        
        # 语义发现
        h1 = F.elu(self.ln1(self.gat1(x, edge_index))) # 第一层 GAT 输入还是原始 x (128维)
        
        # 第二层残差：h_base 结合了第一层语义和初始特征投影
        h_base = self.gat2(h1, edge_index)
        h_base = F.elu(self.ln2(h_base) + x_proj) 
        
        # 递归采样
        weights = self.get_neural_recursive_weights(h_base, edge_index, target_mask, hard=hard)
        
        # 模式增强
        h_enhanced = h_base
        for _ in range(self.hops):
            msg = self.enhancer(h_enhanced, edge_index, weights)
            h_enhanced = h_enhanced + F.elu(msg) 
            
        logits = self.classifier(h_enhanced)
        row, col = edge_index
        pos_edge_feat = torch.cat([h_enhanced[row], h_enhanced[col]], dim=-1)
        link_probs_pos = self.link_predictor(pos_edge_feat).squeeze()
        
        return F.log_softmax(logits, dim=1), link_probs_pos, h_enhanced, weights

# ==========================================
# 数据加载
# ==========================================
dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data', transform=T.ToUndirected())
data = dataset[0]
split_idx = dataset.get_idx_split()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = NeighborLoader(
    data, num_neighbors=[10, 8], batch_size=Config.batch_size,
    input_nodes=split_idx['train'], shuffle=True, num_workers=4, persistent_workers=True
)

model = NeuralRecursiveSystem(dataset.num_features, hidden_size=Config.hidden_dim, out_size=dataset.num_classes, hops=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr_init, weight_decay=Config.weight_decay)
evaluator = Evaluator(name='ogbn-arxiv')

# ==========================================
# 【修改】：带 History 记录的训练逻辑
# ==========================================
def train_minibatch(epoch, total_epochs, start_tau, end_tau):
    model.train()
    total_loss_epoch = 0
    total_phys_edges, total_logic_edges = 0, 0
    all_sp = []

    use_hard = True if epoch > Config.soft_end else False
    curr_tau = max(end_tau, start_tau - (epoch / total_epochs) * (start_tau - end_tau))
    model.tau = curr_tau
    curr_lambda = 0.1 + (epoch/Config.soft_end) * (Config.lambda_max - 0.1) if epoch <= Config.soft_end else Config.lambda_max

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        y_input = batch.y.squeeze().clone()
        y_input[:batch.batch_size] = dataset.num_classes 
        batch_target_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
        batch_target_mask[:batch.batch_size] = True

        log_probs, link_probs_pos, h_enhanced, weights = model(
            batch.x, batch.edge_index, y_input, batch_target_mask, hard=use_hard
        )

        # 指标统计
        total_phys_edges += batch.edge_index.size(1)
        total_logic_edges += (weights > 0.5).sum().item()
        all_sp.append(weights.mean().item())

        loss_clf = F.nll_loss(log_probs[:batch.batch_size], batch.y.squeeze()[:batch.batch_size])
        
        # 链路预测辅助
        neg_edge_index = negative_sampling(batch.edge_index, num_nodes=batch.num_nodes).to(device)
        neg_edge_feat = torch.cat([h_enhanced[neg_edge_index[0]], h_enhanced[neg_edge_index[1]]], dim=-1)
        link_probs_neg = model.link_predictor(neg_edge_feat).squeeze()
        loss_link = (F.binary_cross_entropy(link_probs_pos, torch.ones_like(link_probs_pos)) + \
                     F.binary_cross_entropy(link_probs_neg, torch.zeros_like(link_probs_neg))) * Config.link_weight
        
        loss_sparse = F.mse_loss(weights.mean(), torch.tensor(Config.target_sparsity, device=device))
        
        total_loss = loss_clf + loss_link + curr_lambda * loss_sparse
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss_epoch += total_loss.item()

    avg_loss = total_loss_epoch / len(train_loader)
    avg_sp = np.mean(all_sp)
    pruned_rate = (1 - total_logic_edges / total_phys_edges) * 100
    
    return avg_loss, avg_sp, pruned_rate

# ==========================================
# 可视化工具函数
# ==========================================
def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # 1. Loss & Accuracy
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.plot(range(0, len(history['val_acc'])*5, 5), history['val_acc'], label='Val Acc', color='darkorange', marker='o')
    ax2.set_ylabel('Accuracy')
    ax1.set_title("Training Dynamics"); ax1.legend(loc='upper left'); ax2.legend(loc='upper right')

    # 2. Sparsity
    axes[1].plot(history['sparsity'], label='Actual Sparsity', color='seagreen', linewidth=2)
    axes[1].axhline(y=Config.target_sparsity, color='red', linestyle='--', label='Target')
    axes[1].set_title("Sparsity Convergence"); axes[1].set_xlabel('Epochs'); axes[1].legend()

    # 3. Pruned Rate
    axes[2].plot(history['pruned_rate'], color='crimson', linewidth=2)
    axes[2].set_title("Structural Denoising Rate (%)"); axes[2].set_xlabel('Epochs'); axes[2].set_ylabel('Rate %')
    
    plt.tight_layout(); plt.show()

# ==========================================
# 运行主循环
# ==========================================
history = {'train_loss': [], 'val_acc': [], 'sparsity': [], 'pruned_rate': []}
best_valid_acc = 0.0

print(f"Starting Training on {dataset_name} with Label Reuse...")

for epoch in range(1, Config.epochs + 1):
    avg_loss, avg_sp, p_rate = train_minibatch(epoch, Config.epochs, 1.0, 0.1)
    
    # 记录 History
    history['train_loss'].append(avg_loss)
    history['sparsity'].append(avg_sp)
    history['pruned_rate'].append(p_rate)
    
    if epoch % 5 == 0: 
        model.eval()
        with torch.no_grad():
            eval_data = data.to(device)
            y_eval = eval_data.y.squeeze().clone()
            y_eval[split_idx['valid']] = dataset.num_classes
            y_eval[split_idx['test']] = dataset.num_classes
            
            full_mask = torch.ones(eval_data.num_nodes, dtype=torch.bool, device=device)
            lp, _, _, _ = model(eval_data.x, eval_data.edge_index, y_eval, full_mask, hard=True)
            y_pred = lp.argmax(dim=-1, keepdim=True)
            
            valid_acc = evaluator.eval({'y_true': eval_data.y[split_idx['valid']], 'y_pred': y_pred[split_idx['valid']]})['acc']
            history['val_acc'].append(valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), 'best_arxiv_final.pt')
                print(f"[*] New Best! Val Acc: {valid_acc:.4f} | Pruned: {p_rate:.1f}%")

            del eval_data; torch.cuda.empty_cache()

# 最终展示结果
plot_training_history(history)