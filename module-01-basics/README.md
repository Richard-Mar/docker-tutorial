# Module 01: Docker 基础与核心概念

## 学习目标

完成本模块后，你能够：
- 用"ML 工程师的语言"理解 Docker 的核心抽象
- 拉取、运行、管理容器
- 编写基础 Dockerfile 并构建镜像
- 理解 volumes 和 port mapping

---

## 1. 为什么算法工程师需要 Docker？

### 你现在的痛点

```bash
# 场景：你在本地训练好了模型，发给同事
"你好，运行这个脚本"
"跑不起来，报错: ImportError: No module named 'transformers'"
"你装了 transformers 吗？版本是 4.37.0"
"装了，但我的版本是 4.30.0"
"...那你把 CUDA 版本告诉我"
```

Docker 解决的核心问题：**环境一致性**。

### 用你熟悉的概念类比

| 你已知的概念 | Docker 对应概念 | 说明 |
|------------|----------------|------|
| conda 环境 | Docker Image（镜像）| 冻结的环境快照 |
| 激活的 conda env | Docker Container（容器）| 镜像的运行实例 |
| requirements.txt | Dockerfile | 描述如何构建环境 |
| conda env export | docker image save | 导出环境 |
| PyPI / conda-forge | Docker Hub / Registry | 镜像仓库 |
| Python 进程 | Container 进程 | 隔离运行的程序 |

### Docker vs conda 的关键差异

```
conda 环境：
  - 只隔离 Python 包
  - 共享同一个操作系统
  - 共享同一个 CUDA 驱动

Docker 容器：
  - 隔离整个用户空间（OS、库、工具链）
  - 可以在 Ubuntu 22.04 容器里运行，宿主机是 macOS
  - 可以精确锁定 CUDA 版本（如 12.1.0）
```

---

## 2. 核心概念

### 2.1 Image（镜像）

镜像是**只读的分层文件系统快照**，类比成"打包好的 conda 环境 + 操作系统"。

```
ubuntu:22.04          # 基础层：Ubuntu 操作系统
    └── python:3.11   # 加了 Python
        └── pytorch/pytorch:2.2.0  # 加了 PyTorch
            └── your-training-image  # 你的代码和依赖
```

每一层对应 Dockerfile 里的一条指令（RUN/COPY/ADD）。

### 2.2 Container（容器）

容器是镜像的**可写运行实例**，类比成"激活的 conda 环境 + 运行中的进程"。

```
同一个 Image 可以启动多个 Container：
Image: pytorch-train:v1.0
  ├── Container A: 正在跑 experiment_1
  ├── Container B: 正在跑 experiment_2  
  └── Container C: 停止状态
```

### 2.3 Dockerfile

Dockerfile 是**构建镜像的脚本**，类比成 requirements.txt + conda env create 命令的结合。

### 2.4 Registry（镜像仓库）

存储和分发镜像的服务，类比成 PyPI 或 conda-forge。

- **Docker Hub**：公开仓库，`docker pull pytorch/pytorch`
- **NVIDIA NGC**：NVIDIA 官方 ML 镜像，`nvcr.io/nvidia/pytorch:24.01-py3`
- **私有 Registry**：团队内部使用（AWS ECR / 阿里云 ACR）

---

## 3. 安装 Docker

### macOS（推荐 Docker Desktop）

```bash
# 方法 1：官网下载（推荐）
# https://docs.docker.com/desktop/install/mac-install/

# 方法 2：Homebrew
brew install --cask docker
```

### Linux（Ubuntu/Debian）

```bash
# 官方一键安装脚本
curl -fsSL https://get.docker.com | sh

# 把当前用户加入 docker 组（避免每次 sudo）
sudo usermod -aG docker $USER
newgrp docker

# 验证
docker run hello-world
```

### 验证安装

```bash
docker --version          # Docker version 26.x.x
docker info               # 查看系统信息
docker run hello-world    # 运行第一个容器
```

---

## 4. 基础命令

### 4.1 镜像操作

```bash
# 拉取镜像（类比：conda install）
docker pull python:3.11-slim
docker pull pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 查看本地镜像（类比：conda env list）
docker images
docker image ls

# 删除镜像
docker rmi python:3.11-slim
docker image rm python:3.11-slim

# 搜索镜像
docker search pytorch
```

### 4.2 容器操作

```bash
# 运行容器（类比：conda activate + 执行命令）
docker run python:3.11-slim python -c "import sys; print(sys.version)"

# 交互模式（进入容器的 shell）
docker run -it python:3.11-slim bash

# 后台运行
docker run -d --name my-jupyter jupyter/base-notebook

# 查看运行中的容器
docker ps

# 查看所有容器（包括已停止）
docker ps -a

# 停止/启动/删除容器
docker stop my-jupyter
docker start my-jupyter
docker rm my-jupyter

# 进入已运行的容器
docker exec -it my-jupyter bash

# 查看容器日志
docker logs my-jupyter
docker logs -f my-jupyter   # 实时跟踪（类比 tail -f）
```

### 4.3 数据卷（Volumes）— 重要！

容器内的文件默认在容器删除后消失。ML 场景中**数据集和模型必须挂载**：

```bash
# -v 宿主机路径:容器内路径
docker run -it \
  -v /home/richard/datasets:/workspace/datasets \
  -v /home/richard/models:/workspace/models \
  pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  bash

# 在容器内查看：
ls /workspace/datasets  # 就是宿主机 /home/richard/datasets 的内容
```

### 4.4 端口映射

```bash
# -p 宿主机端口:容器端口
docker run -p 8888:8888 jupyter/base-notebook
# 访问 http://localhost:8888
```

---

## 5. 编写第一个 Dockerfile

### 5.1 基础结构

```dockerfile
# 1. 指定基础镜像
FROM python:3.11-slim

# 2. 设置工作目录（类比：mkdir + cd）
WORKDIR /app

# 3. 复制依赖文件
COPY requirements.txt .

# 4. 安装依赖（类比：pip install -r requirements.txt）
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制代码
COPY . .

# 6. 定义容器启动时执行的命令
CMD ["python", "train.py"]
```

### 5.2 针对 ML 训练的 Dockerfile

```dockerfile
# module-01-basics/exercises/Dockerfile.ml-basic
FROM python:3.11-slim

WORKDIR /workspace

# 先装依赖（利用 Docker 层缓存：依赖不变就不重新安装）
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    numpy \
    scikit-learn \
    matplotlib

# 再复制代码（代码改变不会触发重新安装依赖）
COPY train.py .

CMD ["python", "train.py"]
```

### 5.3 构建和运行

```bash
# 构建（-t 指定镜像名:标签，. 是 Dockerfile 所在目录）
docker build -t my-ml-train:v1.0 .

# 运行
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  my-ml-train:v1.0
```

---

## 6. .dockerignore

类比 `.gitignore`，避免把不必要的文件复制进镜像：

```
# .dockerignore
__pycache__/
*.pyc
*.pth          # 不把模型权重复制进镜像（应挂载）
data/          # 不把数据集复制进镜像（应挂载）
.git/
*.ipynb_checkpoints/
.env
```

---

## 7. 常见错误排查

```bash
# 错误：container 内找不到文件
# → 检查 COPY 路径和 WORKDIR

# 错误：pip 安装失败（网络问题）
# → 换国内镜像源
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...

# 错误：权限不够
# → 检查 volume 挂载的宿主机目录权限
chmod 755 /path/to/your/data

# 镜像太大
# → 使用 slim/alpine 基础镜像，合并 RUN 命令，用 --no-cache-dir
```

---

## 练习

见 `exercises/` 目录。

**Exercise 1:** 用 `python:3.11-slim` 运行一个计算斐波那契数列的脚本  
**Exercise 2:** 为一个 sklearn 训练脚本编写 Dockerfile 并构建镜像  
**Exercise 3:** 挂载本地数据目录，在容器内读取 CSV 文件并输出统计信息

---

## 速查表

见 `cheatsheet.md`。
