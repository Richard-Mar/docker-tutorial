# Module 02: ML 环境容器化

## 学习目标

- 选择正确的 PyTorch/TensorFlow 官方基础镜像
- 为 ML 项目构建生产就绪的镜像（多阶段构建）
- 理解镜像层缓存策略，加速构建
- 在容器中运行 Jupyter Notebook

---

## 1. 选择基础镜像

### 1.1 镜像层次结构

```
ubuntu:22.04  (660MB)
  └── nvidia/cuda:12.1-base-ubuntu22.04  (加 CUDA runtime)
       └── pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime  (加 PyTorch)
            └── your-project-image  (加你的代码)
```

### 1.2 官方镜像选择指南

**PyTorch 官方镜像**（`pytorch/pytorch`）：

| Tag 示例 | 说明 | 适用场景 |
|---------|------|----------|
| `2.2.0-cuda12.1-cudnn8-runtime` | 含 CUDA runtime | **推理/部署** 首选 |
| `2.2.0-cuda12.1-cudnn8-devel` | 含完整 CUDA toolkit | 需要编译 CUDA 扩展 |
| `2.2.0-cuda11.8-cudnn8-runtime` | 旧版 CUDA | 兼容旧硬件 |

**NVIDIA NGC 镜像**（`nvcr.io/nvidia/pytorch`）：

```bash
# NGC 镜像经过 NVIDIA 深度优化，包含 APEX、Transformer Engine 等
nvcr.io/nvidia/pytorch:24.01-py3   # 2024年1月版本
```

**TensorFlow 官方镜像**：

```bash
tensorflow/tensorflow:2.15.0-gpu        # GPU 版
tensorflow/tensorflow:2.15.0-gpu-jupyter # GPU + Jupyter
```

**CPU-only 场景（本地开发、推理服务）**：

```bash
python:3.11-slim              # 最轻量，需自装所有包
pytorch/pytorch:2.2.0         # 官方 CPU-only PyTorch
```

### 1.3 如何选镜像

```
问题 1：需要 GPU 吗？
  ├─ 否 → python:3.11-slim 或 pytorch/pytorch:x.x.x (CPU)
  └─ 是 → 下一步

问题 2：CUDA 版本要求？
  ├─ 用 nvidia-smi 查宿主机驱动支持的最高 CUDA 版本
  └─ 选 ≤ 宿主机支持的 CUDA 版本

问题 3：用途？
  ├─ 训练/开发 → devel 或 NGC 镜像（工具全）
  └─ 推理/部署 → runtime 镜像（体积小）
```

---

## 2. 为训练任务构建镜像

### 2.1 基础版本

```dockerfile
# Dockerfile.train
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 换国内 pip 源（可选）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 先复制依赖文件（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码（代码改变不触发重新安装依赖）
COPY src/ ./src/
COPY configs/ ./configs/

CMD ["python", "src/train.py", "--config", "configs/default.yaml"]
```

```
# requirements.txt
transformers==4.37.0
datasets==2.17.0
accelerate==0.27.0
wandb
tensorboard
tqdm
```

### 2.2 生产级：多阶段构建

多阶段构建用于缩减最终镜像体积（把构建工具和最终运行环境分离）：

```dockerfile
# Dockerfile.production

# ===== 阶段1：构建环境 =====
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel AS builder

WORKDIR /build

# 安装可能需要编译的包（如 flash-attn）
RUN pip install --no-cache-dir \
    flash-attn==2.5.0 \
    --no-build-isolation

# ===== 阶段2：运行环境 =====
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 从构建阶段复制编译好的包
COPY --from=builder /opt/conda/lib/python3.10/site-packages/flash_attn \
     /opt/conda/lib/python3.10/site-packages/flash_attn

# 安装其他运行时依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/train.py"]
```

### 2.3 构建和推送镜像

```bash
# 本地构建
docker build -f Dockerfile.train -t my-pytorch-train:v1.0 .

# 查看镜像大小（优化的重要指标）
docker image ls my-pytorch-train

# 推送到 Docker Hub（需先 docker login）
docker tag my-pytorch-train:v1.0 yourusername/my-pytorch-train:v1.0
docker push yourusername/my-pytorch-train:v1.0

# 推送到私有仓库（如阿里云 ACR）
docker tag my-pytorch-train:v1.0 registry.cn-hangzhou.aliyuncs.com/yournamespace/my-pytorch-train:v1.0
docker push registry.cn-hangzhou.aliyuncs.com/yournamespace/my-pytorch-train:v1.0
```

---

## 3. 层缓存策略（ML 场景的关键技巧）

Docker 构建时，若某层的输入未变化，直接使用缓存。

```dockerfile
# ❌ 错误写法：代码改了，依赖也重装
COPY . .
RUN pip install -r requirements.txt

# ✓ 正确写法：依赖不变不重装
COPY requirements.txt .          # 先只复制依赖文件
RUN pip install -r requirements.txt  # 依赖安装被缓存
COPY . .                         # 代码改变只影响这层之后
```

**ML 场景下的构建顺序建议**：

```dockerfile
# 1. 稳定层（最少改变）→ 最先
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. 系统依赖（改变频率低）
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# 3. Python 依赖（偶尔改变）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 配置文件（有时改变）
COPY configs/ ./configs/

# 5. 源代码（经常改变）→ 最后
COPY src/ ./src/
```

---

## 4. 在容器中运行 Jupyter Notebook

### 4.1 使用官方 Jupyter 镜像

```bash
# 启动 Jupyter（挂载当前目录到 /home/jovyan/work）
docker run --rm -it \
  -p 8888:8888 \
  -v $(pwd):/home/jovyan/work \
  jupyter/pytorch-notebook:latest

# 访问：http://localhost:8888
```

### 4.2 在已有 ML 镜像中添加 Jupyter

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt jupyter jupyterlab

# Jupyter 配置
EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''"]   # 开发环境关闭 token（生产环境不要这么做）
```

```bash
# 运行带 Jupyter 的训练镜像
docker run --rm -it \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  my-pytorch-jupyter:latest
```

---

## 5. 实用技巧

### 5.1 ARG 用于灵活构建

```dockerfile
# 通过 ARG 参数化 PyTorch 版本
ARG PYTORCH_VERSION=2.2.0
ARG CUDA_VERSION=12.1
ARG CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

# 构建时指定版本：
# docker build --build-arg PYTORCH_VERSION=2.1.0 -t my-train:pt210 .
```

### 5.2 ENV 设置常用 ML 环境变量

```dockerfile
# 禁用 Python 输出缓冲（训练日志实时输出）
ENV PYTHONUNBUFFERED=1

# 禁止 Python 生成 .pyc 文件（减少镜像体积）
ENV PYTHONDONTWRITEBYTECODE=1

# 设置 Hugging Face 缓存目录（避免反复下载）
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

# CUDA 相关
ENV CUDA_VISIBLE_DEVICES=0
```

### 5.3 pip 国内镜像加速

```dockerfile
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip install --no-cache-dir -r requirements.txt
```

---

## 练习

**Exercise 1:** 为一个 HuggingFace Transformers 文本分类训练脚本编写 Dockerfile  
**Exercise 2:** 比较使用 `python:3.11-slim` vs `pytorch/pytorch:2.2.0` 作为基础镜像的体积差异  
**Exercise 3:** 实现层缓存优化：修改训练代码后构建，验证依赖不重新安装  
**Exercise 4:** 构建包含 Jupyter 的 PyTorch 镜像并在容器中运行 `.ipynb` 文件
