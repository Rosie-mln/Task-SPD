# Automatic Discovery of Task-Oriented Subgraph Patterns in Large-Scale Graphs

## 1. TSPD Framework 
<!-- Task-SPD 旨在通过 **PANS (Recursive Induction)** 与 **LC (Local Competition)** 协同发现任务特定的结构模式。 -->

<p align="center">
  <img src="overview_01.png" alt="Task-SPD Model Architecture" width="800">
  <br>
  <em>Figure 1: Overview of the Task-SPD Architecture.</em>
</p>

## 2. Code Structure

```text

├── models/
│   ├── backbones/          
│   ├── layers/             
│   │   ├── lc_layer.py     
│   │   └── pans_block.py   
│   └── task_spd.py         
├── engines/                
│   └── co_annealing.py      
├── utils/
│   ├── data_loader/        
│   │   ├── ogb_loader.py
│   │   └── yelp_loader.py
│   └── metrics.py
├── scripts/
│   ├── reproduce/          
│   └── ablation/           
├── configs/                
├── main.py
└── requirements.txt

```



## 3. Python Environment Setup

> [!IMPORTANT]
> **Hardware Support**: All experiments were conducted on NVIDIA RTX A6000 GPUs (48GB VRAM). 

### 🛠 Software Environment
- **OS**: Linux (Ubuntu 20.04/22.04 recommended)
- **Python**: `3.8.10`
- **PyTorch**: `2.4.1`
- **CUDA**: `12.1` (Driver support up to 12.4)
- **PyG**: `2.6.1`

### 🚀 Installation Guide

We recommend using **Conda** to manage dependencies and avoid version conflicts:

```bash
 1. Create and activate the isolated environment
conda create --name TaskSPD python=3.8.10 -y
conda activate TaskSPD

 2. Install PyTorch with specific CUDA 12.1 support
pip install torch==2.4.1 torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

 3. Install Graph Computing Libraries 
pip install torch_geometric==2.6.1
pip install ogb

 4. Install auxiliary tools for configuration and metrics
pip install pyyaml scikit-learns
```


## 4. Data Preparation

This project evaluates the **Task-SPD** framework on three primary benchmark datasets: **ogbn-arxiv**, **ogbn-products**, and **YelpChi**.

### 📂 Local Directory Structure
Please organize the `data/` directory according to the following structure. Note that we use **underscores** (`_`) for directory names to ensure compatibility with our internal data loading logic.

```text
Task-SPD/
└── data/
    ├── ogbn_arxiv/            # OGB 
    │   ├── raw/               # Automatically downloaded files
    │   └── processed/         # Pre-computed graph tensors
    ├── ogbn_products/         # OGB (Large-scale)
    ├── YelpChi/               # YelpChi dataset (Anomaly Detection)
    │   └── raw/
    │       └── YelpChi.mat    # Original MATLAB format file
    ├── Cora/                  # (Optional) Supplementary benchmark
    ├── PubMed/                # (Optional) Supplementary benchmark
    └── CiteSeer/              # (Optional) Supplementary benchmark
```

## 5. Training & Reproduction

### 🚀 Quick Reproduction 
Run the following scripts to reproduce the main results in the paper:

| Target Results | Command | Est. Time (RTX A6000) |
| :--- | :--- | :--- |
| **ogbn-arxiv** | `bash scripts/reproduce/run_arxiv.sh` 
| **ogbn-products** | `bash scripts/reproduce/run_products.sh` 
| **YelpChi** | `bash scripts/reproduce/run_yelp.sh` 

### 🧪 Ablation Study 
To verify the contribution of **PANS** and **LC** modules, use the ablation scripts:

```bash
# Verify w/o PANS (Recursive Induction)
bash scripts/ablation/run_wo_pans.sh

# Verify w/o LC (Local Competition)
bash scripts/ablation/run_wo_lc.sh

python main.py --config [CONFIG_PATH] --model task_spd --backbone [gcn|gat] --device 0
```
## 6. Acknowledgement
This repository is built upon several open-source libraries, including **PyTorch**, **DGL**, and **PyG**. We also thank the **OGB** team for providing the standardized benchmark datasets. The implementation of the co-annealing logic and graph induction modules was inspired by the rigorous experimental standards of the graph machine learning community.