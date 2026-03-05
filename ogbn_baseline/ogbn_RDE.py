# 1.31 版本 baseline - Random DropEdge (Sparsity=0.3 对齐)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import dropout_edge
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# ==========================================
# 1. 实验配置
# ==========================================
class BaseConfig:
    hidden_dim = 128     
    num_heads = 8        
    total_hidden = 1024  # 拼接后 128 * 8 = 1024
    epochs = 150
    patience = 30        
    lr_init = 0.001
    batch_size = 512
    weight_decay = 5e-4
    target_sparsity = 0.30  # 随机保留 30% 的边
    seeds = [1, 7, 314, 1227, 2026]

# ==========================================
# 2. 对齐型 GAT 模型
# ==========================================
class GAT_DropEdge(torch.nn.Module):
    def __init__(self, in_size, head_dim, num_heads, out_size):
        super().__init__()
        # 初始特征投影 (对齐 NRS 的 res_lin)
        self.res_lin = nn.Linear(in_size, head_dim * num_heads)
        
        # GAT 层 (对齐 NRS 的 GAT1 和 GAT2)
        self.conv1 = GATConv(head_dim * num_heads, head_dim, heads=num_heads, dropout=0.5)
        self.bn1 = nn.BatchNorm1d(head_dim * num_heads)
        
        self.conv2 = GATConv(head_dim * num_heads, head_dim, heads=num_heads, dropout=0.5)
        self.bn2 = nn.BatchNorm1d(head_dim * num_heads)
        
        # 分类器 (严格对齐 PANS 的 Sequential 结构)
        self.classifier = nn.Sequential(
            nn.Linear(head_dim * num_heads, (head_dim * num_heads) // 2),
            nn.BatchNorm1d((head_dim * num_heads) // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear((head_dim * num_heads) // 2, out_size)
        )

    def forward(self, x, edge_index):
        x_proj = self.res_lin(x)
        
        # Layer 1
        x = self.conv1(x_proj, edge_index)
        x = F.elu(self.bn1(x))
        
        # Layer 2 + 残差链接
        x = self.conv2(x, edge_index)
        x = F.elu(self.bn2(x)) + x_proj
        
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)

# ==========================================
# 3. Mini-batch 评估函数
# ==========================================
@torch.no_grad()
def inference_minibatch(model, loader, device, evaluator):
    model.eval()
    y_preds, y_trues = [], []
    for batch in loader:
        batch = batch.to(device)
        
        # 推理时也要保持 30% 的稀疏度
        edge_index, _ = dropout_edge(batch.edge_index, p=1 - BaseConfig.target_sparsity)
        
        out = model(batch.x, edge_index)
        y_preds.append(out[:batch.batch_size].argmax(dim=-1).cpu())
        y_trues.append(batch.y[:batch.batch_size].cpu())
        
    y_pred = torch.cat(y_preds, dim=0).numpy().reshape(-1, 1)
    y_true = torch.cat(y_trues, dim=0).numpy().reshape(-1, 1)
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']

# ==========================================
# 4. 实验运行器
# ==========================================
def run_experiment(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
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

    model = GAT_DropEdge(dataset.num_features, BaseConfig.hidden_dim, BaseConfig.num_heads, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BaseConfig.lr_init, weight_decay=BaseConfig.weight_decay)

    best_val_acc, final_test_acc, stagnant_count = 0, 0, 0

    for epoch in range(1, BaseConfig.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 每一轮训练都随机删除 70% 的边
            edge_index, _ = dropout_edge(batch.edge_index, p=1 - BaseConfig.target_sparsity)
            
            out = model(batch.x, edge_index)
            loss = F.nll_loss(out[:batch.batch_size], batch.y.squeeze()[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            val_acc = inference_minibatch(model, valid_loader, device, evaluator)
            test_acc = inference_minibatch(model, test_loader, device, evaluator)
            
            if val_acc > best_val_acc:
                best_val_acc, final_test_acc, stagnant_count = val_acc, test_acc, 0
                status = "[*] NEW BEST"
            else:
                stagnant_count += 5
                status = "[ ]"
            
            print(f"Seed: {seed:4d} | Ep: {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} {status}")
            if stagnant_count >= BaseConfig.patience: break
            
    return final_test_acc

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    print(f"Starting Baseline: GAT + Random DropEdge (Target Sparsity: {BaseConfig.target_sparsity})")
    for s in BaseConfig.seeds:
        acc = run_experiment(s, device)
        results.append(acc)
    
    print("\n" + "="*40)
    print(f"Final GAT-DropEdge Results: {np.mean(results)*100:.2f} ± {np.std(results)*100:.2f}")
    print("="*40)