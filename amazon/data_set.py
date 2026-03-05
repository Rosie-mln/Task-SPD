import torch
import numpy as np
import scipy.io as sio
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import homophily, from_scipy_sparse_matrix
import torch_geometric.transforms as T

def analyze_ogb(name):
    """统计 OGB 数据集"""
    print(f"\n[Dataset: {name}] Analyzing...")
    dataset = PygNodePropPredDataset(name=name, root='data', transform=T.ToUndirected())
    data = dataset[0]
    
    y = data.y.squeeze()
    # 计算图片公式定义的 Edge Homophily
    h_edge = homophily(data.edge_index, y, method='edge')
    
    num_nodes = data.num_nodes
    num_edges = data.num_edges // 2 # 还原为无向边数
    avg_deg = (2 * num_edges) / num_nodes
    
    return [num_nodes, num_edges, avg_deg, dataset.num_classes, h_edge]

def analyze_yelpchi(file_path=r'/home/malina/mln_work/GAT+Sample/amazon/data/YelpChi/raw/YelpChi.mat'):
    """统计 YelpChi 数据集 (异质图)"""
    print(f"\n[Dataset: YelpChi] Analyzing...")
    if not torch.os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please ensure the .mat file is in data folder.")
        return None
    
    # 加载 .mat 文件
    mat = sio.loadmat(file_path)
    # YelpChi 通常包含 net_rur, net_rsr, net_rtr 三种对称矩阵
    # 我们将它们合并以计算整体结构的统计特性
    adj = mat['net_rur'] + mat['net_rsr'] + mat['net_rtr']
    y = torch.tensor(mat['label'].flatten()).long()
    
    edge_index, _ = from_scipy_sparse_matrix(adj)
    
    num_nodes = y.size(0)
    # 去重并转为无向边统计 (scipy matrix 已经是对称的)
    num_edges = edge_index.size(1) // 2 
    avg_deg = (2 * num_edges) / num_nodes
    
    # 计算 Edge Homophily
    h_edge = homophily(edge_index, y, method='adjusted')
    
    return [num_nodes, num_edges, avg_deg, 2, h_edge] # YelpChi 是 2 分类

def print_latex_row(name, stats):
    if stats:
        # 格式: Name & |V| & |E| & Avg.Deg & Classes & h & Metric
        v, e, deg, c, h = stats
        metric = "Acc." if "ogbn" in name else "AUC/F1"
        print(f"\\textit{{{name}}} & {v:,} & {e:,} & {deg:.1f} & {c} & {h:.2f} & {metric} \\\\")

if __name__ == "__main__":
    results = {}
    
    # 1. 统计 OGB 数据集
    for name in ['ogbn-arxiv', 'ogbn-products']:
        try:
            results[name] = analyze_ogb(name)
        except Exception as err:
            print(f"Failed to analyze {name}: {err}")

    # 2. 统计 YelpChi (请确保 data/YelpChi.mat 路径正确)
    try:
        results['YelpChi'] = analyze_yelpchi()
    except Exception as err:
        print(f"Failed to analyze YelpChi: {err}")

    # 3. 输出 LaTeX 表格行
    print("\n" + "="*30)
    print("GENERATED LATEX ROWS:")
    print("="*30)
    for name in ['ogbn-arxiv', 'ogbn-products', 'YelpChi']:
        if name in results:
            print_latex_row(name, results[name])