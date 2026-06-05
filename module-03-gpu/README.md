# Module 03: GPU 容器与深度学习

## 学习目标

- 安装和配置 NVIDIA Container Toolkit
- 在容器中访问 GPU 并验证 CUDA 可用性
- 使用 NVIDIA NGC 官方优化镜像
- 多 GPU 训练的容器配置

---

## 1. NVIDIA Container Toolkit 原理

```
宿主机:  NVIDIA Driver (≥ 525.xx)
              ↓
         nvidia-container-toolkit
              ↓
         Docker Engine
              ↓
容器内:  CUDA Runtime (由镜像提供)
         cuDNN
         PyTorch / TensorFlow
```

**关键理解**：
- 宿主机只需安装 NVIDIA 驱动，**不需要**安装 CUDA
- CUDA toolkit 由 Docker 镜像提供
- 容器内的 CUDA 版本 ≤ 宿主机驱动支持的最高 CUDA 版本

### 检查宿主机支持的最高 CUDA 版本

```bash
nvidia-smi  # 右上角显示 "CUDA Version: XX.X"
```

---

## 2. 安装 NVIDIA Container Toolkit

### Ubuntu/Debian

```bash
# 1. 添加 NVIDIA 仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. 验证
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 3. GPU 容器基础操作

### 3.1 --gpus 参数

```bash
# 使用所有 GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 使用指定数量的 GPU
docker run --rm --gpus 2 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime python -c \
  "import torch; print(torch.cuda.device_count())"

# 使用特定 GPU（按设备 ID）
docker run --rm --gpus '"device=0,1"' pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime nvidia-smi

# 使用单个 GPU
docker run --rm --gpus '"device=0"' pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime nvidia-smi
```

### 3.2 验证 GPU 在容器内可用

```bash
docker run --rm --gpus all \
  pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('GPU name:', torch.cuda.get_device_name(0))
print('CUDA version:', torch.version.cuda)
"
```

---

## 4. 构建 GPU 训练镜像

### 4.1 标准 GPU 训练 Dockerfile

```dockerfile
# Dockerfile.gpu-train
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 源代码
COPY src/ ./src/

CMD ["python", "src/train.py"]
```

### 4.2 运行 GPU 训练容器

```bash
docker run --rm \
  --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/configs:/workspace/configs \
  my-gpu-train:latest \
  python src/train.py --config configs/train.yaml
```

---

## 5. 使用 NVIDIA NGC 镜像

NGC 镜像经过 NVIDIA 深度优化，性能优于官方 PyTorch 镜像，是**生产训练推荐**选项。

```bash
# 查看可用版本：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# 格式：nvcr.io/nvidia/pytorch:YY.MM-py3

# 拉取最新版（以 24.01 为例）
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# NGC 镜像包含：
# - PyTorch（NVIDIA 优化版）
# - CUDA + cuDNN + NCCL
# - Apex（混合精度训练）
# - Transformer Engine（FP8 训练）
# - OpenMPI（分布式训练）
```

```dockerfile
# 基于 NGC 的 Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace

COPY requirements.txt .
# NGC 镜像已有 PyTorch，只装额外依赖
RUN pip install --no-cache-dir \
    transformers==4.37.0 \
    datasets==2.17.0 \
    accelerate==0.27.0 \
    wandb

COPY src/ ./src/
```

---

## 6. 多 GPU 训练

### 6.1 DDP（单机多卡）

```bash
# 容器内运行 DDP 训练（指定 4 个 GPU）
docker run --rm \
  --gpus '"device=0,1,2,3"' \
  -v $(pwd):/workspace \
  my-gpu-train:latest \
  torchrun --nproc_per_node=4 src/train_ddp.py
```

### 6.2 共享内存（--shm-size）

DDP 训练使用共享内存，默认 64MB 可能不够：

```bash
docker run --rm \
  --gpus all \
  --shm-size=8g \            # 增加共享内存
  -v $(pwd):/workspace \
  my-gpu-train:latest \
  torchrun --nproc_per_node=4 src/train_ddp.py
```

### 6.3 网络模式（多机多卡）

```bash
# 使用 host 网络（减少 NCCL 通信开销）
docker run --rm \
  --gpus all \
  --network host \
  --shm-size=8g \
  my-gpu-train:latest \
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 \
    src/train_ddp.py
```

---

## 7. 实用监控命令

```bash
# 容器内实时 GPU 监控
docker exec -it <container> nvidia-smi dmon -s pucvmet

# 容器内查看 GPU 内存
docker exec -it <container> python -c \
  "import torch; print(torch.cuda.memory_summary())"

# 宿主机监控所有 GPU（含容器）
watch -n 1 nvidia-smi
```

---

## 8. macOS 注意事项

macOS 不支持 NVIDIA GPU（Apple Silicon 是 MPS，不是 CUDA）。

```bash
# Apple Silicon 使用 MPS 后端
docker run --rm pytorch/pytorch:2.2.0 python -c \
  "import torch; print(torch.backends.mps.is_available())"
# 但容器内无法访问 MPS，macOS 上开发建议直接用 conda 环境
# Docker 在 macOS 主要用于：测试部署流程、CI、非 GPU 任务
```

---

## 练习

**Exercise 1:** 安装 NVIDIA Container Toolkit，在容器内运行 `nvidia-smi`  
**Exercise 2:** 运行一个简单的 PyTorch GPU 张量运算，验证 CUDA 可用  
**Exercise 3:** 用 NGC 镜像跑一个 ResNet 训练脚本，对比与普通 PyTorch 镜像的速度差异  
**Exercise 4:** 配置 DDP 训练，在 2 张 GPU 上运行分布式训练
