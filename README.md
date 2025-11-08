# Transformer 实现与消融实验

本项目从零实现了一个简化版的 Transformer 模型，并在 Tiny Shakespeare 数据集上进行了训练和消融实验。项目包括 Scaled Dot-Product Attention、多头注意力、位置编码、前馈网络、残差连接和 LayerNorm 等核心组件的实现。

## 目录结构

```
.
├── src/                    # 源代码目录
│   ├── data.py             # 数据处理模块
│   ├── model.py            # 模型实现
│   ├── train.py            # 训练脚本
│   └── utils.py            # 工具函数
├── requirements.txt        # 依赖包列表
├── scripts/
│   ├── run.sh              # 自动化运行脚本
│   └── doc-sh.md           # 脚本说明
├── results/                # 实验结果目录
└── README.md               # 本文件
```

## 硬件要求

- **最低配置**：

  - CPU: 2核以上
  - 内存: 4GB以上
  - 硬盘空间: 1GB以上可用空间
- **推荐配置**：

  - GPU: CUDA 兼容 GPU (如 NVIDIA GTX 1060 或更高)
  - 内存: 8GB以上
  - 硬盘空间: 2GB以上可用空间
- **操作系统**：

  - Windows 10/11
  - Linux (Ubuntu 18.04 或更高版本)
  - macOS (10.14 或更高版本)

## 环境依赖

- Python 3.7+
- PyTorch >= 1.12.0
- numpy
- matplotlib
- tqdm
- requests

安装依赖包：

```bash
pip install -r requirements.txt
```

## 运行说明

### 自动运行所有实验

使用提供的脚本自动运行所有实验：

```bash
bash scripts/run.sh
```

在 Windows 系统上，可以使用 PowerShell 逐条执行命令。

### 手动运行实验

#### 基线模型 (位置编码 + 相对位置偏置)

```bash
python src/train.py --task seq2seq --data data/tiny_shakespeare.txt --seq_len 128 --batch_size 32 --epochs 6 --use_pos_encoding --relative_pos --save results/seq2seq_base --seed 42
```

#### 消融实验 1：无位置编码

```bash
python src/train.py --task seq2seq --data data/tiny_shakespeare.txt --seq_len 128 --batch_size 32 --epochs 6 --save results/no_pos --seed 42
```

#### 消融实验 2：单头注意力

```bash
python src/train.py --task seq2seq --data data/tiny_shakespeare.txt --seq_len 128 --batch_size 32 --epochs 6 --heads 1 --use_pos_encoding --save results/one_head --seed 42
```

#### 消融实验 3：无残差连接

```bash
python src/train.py --task seq2seq --data data/tiny_shakespeare.txt --seq_len 128 --batch_size 32 --epochs 6 --use_pos_encoding --no_residual --save results/no_residual --seed 42
```

## 重现实验

为了确保实验的可重现性，所有训练命令都使用了固定的随机种子 (`--seed 42`)。如果需要重现实验，请使用上述命令中的 exact 参数配置。

每个实验的结果将保存在 `results/` 目录下的相应子目录中，包括：

- 模型检查点 (model_epoch*.pt)
- 训练配置 (train_config.json)
- 词汇表 (vocab.json)
- 训练损失曲线 (train_loss.png)

## 项目特点

- 字符级语言建模
- 支持 GPU 自动检测与使用 (CUDA)
- 完整的训练流程和可视化
- 消融实验验证各组件作用
- 可复现实验 (通过固定随机种子)
