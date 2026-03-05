# 1.31 ogbn-arxiv baseline top-k pruning


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
    weight_decay = 5e-4
    target_sparsity = 0.30  
    seeds = [1, 7, 314, 1227, 2026]

# ==========================================
# 2. 剪枝模型
# ==========================================
class GAT_Pruning(torch.nn.Module):
    def __init__(self, in_size, head_dim, num_heads, out_size):
        super().__init__()
        self.res_lin = nn.Linear(in_size, head_dim * num_heads)
        
        # 对齐 NRS 的 GAT 
        self.conv1 = GATConv(head_dim * num_heads, head_dim, heads=num_heads, dropout=0.5)
        self.bn1 = nn.BatchNorm1d(head_dim * num_heads)
        
        self.conv2 = GATConv(head_dim * num_heads, head_dim, heads=num_heads, dropout=0.5)
        self.bn2 = nn.BatchNorm1d(head_dim * num_heads)
        
        self.classifier = nn.Sequential(
            nn.Linear(head_dim * num_heads, (head_dim * num_heads) // 2),
            nn.BatchNorm1d((head_dim * num_heads) // 2),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear((head_dim * num_heads) // 2, out_size)
        )

    def prune_edges(self, edge_index, edge_attr, sparsity):
        """
        基于attention权重进行 top-k 剪枝
        """
        if edge_attr is None or edge_index.numel() == 0:
            return edge_index

        score = edge_attr.mean(dim=-1)
        num_edges = edge_index.size(1)
        
        # 2. 确保 1 <= k <= num_edges，防止 k=0 崩溃
        k = int(num_edges * sparsity)
        k = max(1, min(k, num_edges))
        
        # 3. 获取索引
        _, top_indices = torch.topk(score, k, sorted=False)
        return edge_index[:, top_indices]

    def forward(self, x, edge_index):
        x_proj = self.res_lin(x)
        

        x, (edge_index_tmp, alpha) = self.conv1(x_proj, edge_index, return_attention_weights=True)
        x = F.elu(self.bn1(x))
        
        pruned_edge_index = self.prune_edges(edge_index_tmp, alpha, BaseConfig.target_sparsity)
        
        # 第二层聚合
        x = self.conv2(x, pruned_edge_index)
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
        out = model(batch.x, batch.edge_index)
        y_preds.append(out[:batch.batch_size].argmax(dim=-1).cpu())
        y_trues.append(batch.y[:batch.batch_size].cpu())
        
    y_pred = torch.cat(y_preds, dim=0).numpy().reshape(-1, 1)
    y_true = torch.cat(y_trues, dim=0).numpy().reshape(-1, 1)
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']

# ==========================================
# 4. 实验运行
# ==========================================
def run_experiment(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data', transform=T.ToUndirected())
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-arxiv')

    train_loader = NeighborLoader(data, num_neighbors=[20, 15], batch_size=BaseConfig.batch_size,
                                 input_nodes=split_idx['train'], shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=BaseConfig.batch_size,
                                 input_nodes=split_idx['valid'], shuffle=False)
    test_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=BaseConfig.batch_size,
                                input_nodes=split_idx['test'], shuffle=False)

    model = GAT_Pruning(dataset.num_features, BaseConfig.hidden_dim, BaseConfig.num_heads, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BaseConfig.lr_init, weight_decay=BaseConfig.weight_decay)

    best_val_acc, final_test_acc, stagnant_count = 0, 0, 0

    for epoch in range(1, BaseConfig.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
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
            if stagnant_count >= BaseConfig.patience: break
            
    return final_test_acc

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    print(f"Starting Baseline: GAT + Top-K Pruning (Target Sparsity: {BaseConfig.target_sparsity})")
    for s in BaseConfig.seeds:
        acc = run_experiment(s, device)
        results.append(acc)
    
    print("\n" + "="*40)
    print(f"Final GAT-Pruning Results: {np.mean(results)*100:.2f} ± {np.std(results)*100:.2f}")
    print("="*40) 