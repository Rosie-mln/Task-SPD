# 1.31 版本 baseline -  (Vanilla Feature-based GCN)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# ==========================================
# 1. 实验配置
# ==========================================
class BaseConfig:
    hidden_dim = 128
    total_hidden = 1024  
    epochs = 150
    patience = 30        
    lr_init = 0.001
    batch_size = 512
    weight_decay = 5e-4
    seeds = [1, 7, 314, 1227, 2026]

# ==========================================
# 2. 对齐 GCN 模型
# ==========================================
class GCN_Vanilla(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # 保留特征映射
        self.res_lin = nn.Linear(in_size, hidden_size)
        
        # GCN 层
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, out_size)
        )

    def forward(self, x, edge_index):
        # 初始特征投影
        x_proj = self.res_lin(x)
        x = x_proj
        
        # Conv 1
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        # Conv 2 + 残差链接 (特征残差)
        x = F.relu(self.bn2(self.conv2(x, edge_index))) + x_proj
        
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
        out = model(batch.x, batch.edge_index)
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
    torch.backends.cudnn.deterministic = True
    
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

    model = GCN_Vanilla(dataset.num_features, BaseConfig.total_hidden, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BaseConfig.lr_init, weight_decay=BaseConfig.weight_decay)

    best_val_acc, final_test_acc, stagnant_count = 0, 0, 0

    for epoch in range(1, BaseConfig.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 直接前向传播，不使用标签输入
            out = model(batch.x, batch.edge_index)
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
            if stagnant_count >= BaseConfig.patience: 
                break
            
    return final_test_acc

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    print("-" * 50)
    print(f"Starting Baseline: Vanilla GCN (No Label Reuse) ({len(BaseConfig.seeds)} Seeds)")
    print("-" * 50)

    for i, s in enumerate(BaseConfig.seeds):
        print(f"\n[Run {i+1}/{len(BaseConfig.seeds)}] Seed: {s}")
        acc = run_experiment(s, device)
        results.append(acc)
    
    mean_res = np.mean(results) * 100
    std_res = np.std(results) * 100
    
    print("\n" + "=" * 50)
    print("FINAL VANILLA GCN RESULTS")
    print("-" * 50)
    print(f"Mean Accuracy: {mean_res:.2f}%")
    print(f"Std Deviation: {std_res:.2f}%")
    print(f"LaTeX Format: {mean_res:.2f} \pm {std_res:.2f}")
    print("=" * 50)