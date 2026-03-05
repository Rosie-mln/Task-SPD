# 1.31更新 ogbn baseline NeuralSparse 


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# ==========================================
# 1. 实验配置
# ==========================================
class BaseConfig:
    hidden_dim = 128     
    num_heads = 8        
    total_hidden = 1024  
    epochs = 150
    patience = 30        
    lr_init = 0.001
    batch_size = 512
    target_sparsity = 0.30  
    weight_decay = 5e-4
    seeds = [1, 7, 314, 1227, 2026]

# ==========================================
# 2. NeuralSparse 
# ==========================================
class NeuralSparse_Sampler(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # 学习边的评分逻辑
        self.scorer = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, 1)
        )

    def forward(self, h, edge_index, tau=1.0, hard=True):
        row, col = edge_index
        edge_feat = torch.cat([h[row], h[col]], dim=-1)
        logits = self.scorer(edge_feat).squeeze()
        
        # 二分类采样空间：[Discard, Keep]
        sampling_logits = torch.stack([torch.zeros_like(logits), logits], dim=-1)
        
        # Gumbel-Softmax 重参数化采样
        weights = F.gumbel_softmax(sampling_logits, tau=tau, hard=hard, dim=-1)[:, 1]
        return weights, logits

class NeuralSparse_System(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.total_hidden = hidden_size * 8
        self.res_lin = nn.Linear(in_size, self.total_hidden)
        
        self.gat1 = GATConv(self.total_hidden, hidden_size, heads=8, dropout=0.5)
        self.bn1 = nn.BatchNorm1d(self.total_hidden)
        self.gat2 = GATConv(self.total_hidden, hidden_size, heads=8, dropout=0.5)
        self.bn2 = nn.BatchNorm1d(self.total_hidden)
        
        self.sampler_net = NeuralSparse_Sampler(self.total_hidden)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.total_hidden, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_size * 4, out_size)
        )

    def forward(self, x, edge_index, tau=1.0, hard=True):
        x_proj = self.res_lin(x)
        
        # 1. 特征编码
        h1 = F.elu(self.bn1(self.gat1(x_proj, edge_index)))
        h_base = F.elu(self.bn2(self.gat2(h1, edge_index)))
        
        # 2. NeuralSparse 采样 (单步，非递归)
        weights, logits_raw = self.sampler_net(h_base, edge_index, tau=tau, hard=hard)
        
        # 3. 稀疏聚合
        row, col = edge_index
        h_sparse = h_base + (weights.view(-1, 1) * h_base[col]).new_zeros(h_base.shape).index_add_(0, row, weights.view(-1, 1) * h_base[col])
        
        logits = self.classifier(h_sparse)
        return F.log_softmax(logits, dim=1), weights, logits_raw

# ==========================================
# 3. 训练与评估逻辑
# ==========================================
def train_step(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    # Tau 退火：模仿 PANS 的训练过程
    tau = max(0.2, 1.0 - (epoch / 50)) 
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        log_probs, weights, _ = model(batch.x, batch.edge_index, tau=tau, hard=(epoch > 50))
        
        # 分类损失
        loss_clf = F.nll_loss(log_probs[:batch.batch_size], batch.y.squeeze()[:batch.batch_size])
        
        # 稀疏度约束：拉近平均权重与 target_sparsity
        loss_sp = F.l1_loss(weights.mean(), torch.tensor(BaseConfig.target_sparsity, device=device))
        
        loss = loss_clf + 1.0 * loss_sp # 1.0 为 Sparsity Penalty 权重
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, evaluator):
    model.eval()
    y_preds, y_trues = [], []
    for batch in loader:
        batch = batch.to(device)
        lp, _, _ = model(batch.x, batch.edge_index, hard=True)
        y_preds.append(lp[:batch.batch_size].argmax(dim=-1).cpu())
        y_trues.append(batch.y[:batch.batch_size].cpu())
        
    y_pred = torch.cat(y_preds, dim=0).numpy().reshape(-1, 1)
    y_true = torch.cat(y_trues, dim=0).numpy().reshape(-1, 1)
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']

# ==========================================
# 4. 实验主循环
# ==========================================
def run_experiment(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data', transform=T.ToUndirected())
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-arxiv')

    train_loader = NeighborLoader(data, num_neighbors=[20, 15], batch_size=BaseConfig.batch_size,
                                 input_nodes=split_idx['train'], shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=BaseConfig.batch_size,
                                 input_nodes=split_idx['valid'], shuffle=False)
    test_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=BaseConfig.batch_size,
                                input_nodes=split_idx['test'], shuffle=False)

    model = NeuralSparse_System(dataset.num_features, BaseConfig.hidden_dim, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BaseConfig.lr_init, weight_decay=BaseConfig.weight_decay)

    best_val, final_test, stagnant = 0, 0, 0
    for epoch in range(1, BaseConfig.epochs + 1):
        loss = train_step(model, train_loader, optimizer, device, epoch)
        
        if epoch % 5 == 0:
            val_acc = evaluate(model, valid_loader, device, evaluator)
            test_acc = evaluate(model, test_loader, device, evaluator)
            
            if val_acc > best_val:
                best_val, final_test, stagnant = val_acc, test_acc, 0
                status = "[*] NEW BEST"
            else:
                stagnant += 5; status = "[ ]"
            
            print(f"Seed: {seed} | Ep: {epoch:03d} | Val: {val_acc:.4f} | Test: {test_acc:.4f} {status}")
            if stagnant >= BaseConfig.patience: break
            
    return final_test

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    print(f"Starting Baseline: NeuralSparse (Target Sparsity: {BaseConfig.target_sparsity})")
    for s in BaseConfig.seeds:
        acc = run_experiment(s, device)
        results.append(acc)
    
    print(f"\nFinal NeuralSparse Results: {np.mean(results)*100:.2f} ± {np.std(results)*100:.2f}")