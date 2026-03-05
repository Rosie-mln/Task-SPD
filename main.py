import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import os

import os.path as osp
import torch
from torch_geometric.datasets import Planetoid

# 【数据加载】
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Cora')

# 手动实例化但不触发 download
# 我们先定义 dataset，然后通过 transform 强行加载本地已经存在的数据
dataset = Planetoid(root=path, name='Cora')


data = dataset[0]
print(f'成功加载！节点数: {data.num_nodes}')


print(f'数据集信息: 节点数 {data.num_nodes}, 边数 {data.num_edges}')
print(f'特征维度: {dataset.num_features}, 类别数: {dataset.num_classes}')

# 【2. 模型定义】
class GAT(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(GAT, self).__init__()
        # 第一层：输入特征 -> 隐藏层 (使用 8 个注意力头)
        self.conv1 = GATConv(in_size, hidden_size, heads=8, dropout=0.6)
        # 第二层：多头拼接后的维度是 hidden_size * 8 -> 最终分类数
        self.conv2 = GATConv(hidden_size * 8, out_size, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # ELU 是常用的激活函数，类似 ReLU 但在负数区域更平滑
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # 返回对数概率，用于计算损失
        return F.log_softmax(x, dim=1)

# 【3. 训练准备】
model = GAT(in_size=dataset.num_features, hidden_size=8, out_size=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
losses = []

# 【4. 训练与评价函数】
def test():
    model.eval() # 切换到评估模式（关闭 Dropout）
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1) # 找到概率最大的类别
    
    # 分别计算训练集、验证集和测试集的准确率
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        accs.append(int(correct.sum()) / int(mask.sum()))
    return accs

# 【5. 启动训练循环】
print("\n开始训练...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    # 前向传播
    out = model(data.x, data.edge_index)
    # 计算损失 (只针对有标签的训练节点)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # 反向传播与优化
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')

# 【6. 结果查看：损失曲线】
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 【7. 查看具体的分类结果】
model.eval()
logits = model(data.x, data.edge_index)
preds = logits.argmax(dim=1)
print(f"\n前5个节点的预测类别: {preds[:5].tolist()}")
print(f"前5个节点的真实类别: {data.y[:5].tolist()}")