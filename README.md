# Docker for ML Engineers — 算法工程师的 Docker 实战教程

## 定位与目标

本教程面向**有 Python/PyTorch/TF 经验但 Docker 零基础**的算法工程师。  
目标：能够独立完成 ML 项目的容器化、GPU 训练环境搭建、模型服务部署和 MLOps 流水线构建。

## 先修条件

- Python 熟练（能写训练脚本）
- 用过 conda 或 virtualenv 管理环境
- 会基础 Linux 命令（ls, cd, vim/nano, ps, chmod）
- 有 ML/DL 项目经验

## 学习路径（预计 3-4 周）

```
Week 1: Module 01 + Module 02
Week 2: Module 03 + Module 04
Week 3: Module 05
Week 4: Module 06 (综合项目)
```

## 模块总览

| 模块 | 主题 | 核心技能 | 预计耗时 |
|------|------|----------|----------|
| 01 | Docker 基础与核心概念 | image/container/Dockerfile | 3-4h |
| 02 | ML 环境容器化 | 替代 conda，PyTorch/TF 镜像 | 3-4h |
| 03 | GPU 容器与深度学习 | NVIDIA Container Toolkit | 3-4h |
| 04 | 模型服务化 | FastAPI + Docker 推理服务 | 4-5h |
| 05 | Docker Compose 编排 | 多服务 ML 平台 | 4-5h |
| 06 | MLOps 实战 | MLflow + CI/CD + 生产部署 | 6-8h |

## 目录结构

```
docker/
├── README.md                  # 本文件
├── module-01-basics/          # Docker 基础
│   ├── README.md
│   ├── exercises/
│   └── cheatsheet.md
├── module-02-ml-env/          # ML 环境容器化
├── module-03-gpu/             # GPU 容器
├── module-04-serving/         # 模型推理服务
├── module-05-compose/         # Compose 编排
└── module-06-mlops/           # MLOps 实战
```

## 推荐参考资源

### 免费
- [MLOps Zoomcamp](https://datatalks.club/blog/mlops-zoomcamp.html) — DataTalks.Club，含 Docker 模块
- [Docker 官方文档](https://docs.docker.com/) — 查手册首选
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — GPU 必读

### 付费（可选）
- Udemy: *Ultimate Docker Bootcamp for ML, GenAI, and Agentic AI*
- Coursera: *MLOps and LLMOps: Deploying and Scaling AI in Production*

### 官方镜像
- PyTorch: `pytorch/pytorch` on Docker Hub
- TensorFlow: `tensorflow/tensorflow` on Docker Hub  
- NVIDIA NGC: `nvcr.io/nvidia/pytorch:xx.xx-py3`
