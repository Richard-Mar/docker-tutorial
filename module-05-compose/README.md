# Module 05: Docker Compose 多服务编排

## 学习目标

- 用 Docker Compose 管理多服务 ML 系统
- 构建 MLflow 实验追踪平台
- 实现服务依赖和健康检查
- 搭建本地 ML 开发平台

---

## 1. 为什么需要 Docker Compose？

ML 系统往往不只一个服务：

```
一个完整的 ML 平台可能包括：
  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
  │ 训练容器    │  │ 推理服务     │  │ Jupyter Lab   │
  │ (PyTorch)   │  │ (FastAPI)    │  │ (开发环境)    │
  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘
         │                │                  │
  ┌──────▼──────┐  ┌──────▼───────┐  ┌───────▼───────┐
  │ MLflow      │  │ PostgreSQL   │  │ MinIO         │
  │ (实验追踪) │  │ (元数据存储) │  │ (模型仓库)   │
  └─────────────┘  └──────────────┘  └───────────────┘
```

Docker Compose 用**一个 YAML 文件**定义和管理所有这些服务。

---

## 2. Docker Compose 基础语法

```yaml
# docker-compose.yml
version: "3.8"

services:
  service-name:
    image: some-image:tag      # 使用现有镜像
    # 或
    build:                     # 从 Dockerfile 构建
      context: .
      dockerfile: Dockerfile.train

    ports:
      - "宿主机端口:容器端口"
    
    volumes:
      - "./host/path:/container/path"
      - named-volume:/data       # 命名卷
    
    environment:
      - KEY=VALUE
    
    env_file:
      - .env                     # 从文件读取环境变量
    
    depends_on:                  # 依赖其他服务先启动
      db:
        condition: service_healthy
    
    networks:
      - ml-network
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  named-volume:                  # 声明命名卷

networks:
  ml-network:                    # 声明网络（服务间可互相访问）
```

---

## 3. 实战：本地 MLflow 实验追踪平台

### 3.1 平台架构

```
mlflow-ui (port 5000)
  └── 读写 → postgres (元数据)
  └── 读写 → minio (模型 artifacts)

训练脚本 → mlflow.log_metric() → mlflow-ui
```

### 3.2 docker-compose.yml

```yaml
# module-05-compose/mlflow-platform/docker-compose.yml
version: "3.8"

services:
  # ===== PostgreSQL：存储实验元数据 =====
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow123
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ml-network

  # ===== MinIO：存储模型文件（S3 兼容）=====
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"    # S3 API
      - "9001:9001"    # Web Console
    volumes:
      - minio-data:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - ml-network

  # ===== MinIO 初始化（创建 bucket）=====
  minio-init:
    image: minio/mc:latest
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 minioadmin minioadmin123;
      mc mb --ignore-existing local/mlflow;
      echo 'Bucket created';
      "
    networks:
      - ml-network

  # ===== MLflow Tracking Server =====
  mlflow:
    image: python:3.11-slim
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow123@postgres:5432/mlflow
      --default-artifact-root s3://mlflow/
      --host 0.0.0.0
      --port 5000
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin123
    ports:
      - "5000:5000"
    depends_on:
      postgres:
        condition: service_healthy
      minio-init:
        condition: service_completed_successfully
    networks:
      - ml-network

  # ===== Jupyter Lab：开发环境 =====
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile.jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/workspace/notebooks
      - ./data:/workspace/data
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin123
    depends_on:
      - mlflow
    networks:
      - ml-network

volumes:
  postgres-data:
  minio-data:

networks:
  ml-network:
    driver: bridge
```

### 3.3 Dockerfile.jupyter

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    jupyterlab \
    mlflow \
    boto3 \
    scikit-learn \
    matplotlib \
    pandas

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", "--NotebookApp.token=''"]
```

### 3.4 使用平台

```bash
# 启动所有服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f mlflow
docker compose logs -f jupyter

# 访问服务
# MLflow UI: http://localhost:5000
# Jupyter:   http://localhost:8888
# MinIO:     http://localhost:9001

# 在训练脚本中使用 MLflow
```

```python
# notebooks/train_with_mlflow.py
import mlflow
import os

# 自动从环境变量读取 MLFLOW_TRACKING_URI
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    # 记录超参数
    mlflow.log_params({"lr": 0.001, "batch_size": 32, "epochs": 10})
    
    for epoch in range(10):
        loss = train_one_epoch(...)
        acc = evaluate(...)
        
        # 记录指标
        mlflow.log_metrics({"train_loss": loss, "val_acc": acc}, step=epoch)
    
    # 保存模型
    mlflow.pytorch.log_model(model, "model")
```

---

## 4. 常用 Compose 命令

```bash
# 启动（-d 后台运行）
docker compose up -d

# 启动指定服务
docker compose up -d mlflow postgres

# 停止（不删除容器）
docker compose stop

# 停止并删除容器（保留 volumes）
docker compose down

# 停止并删除容器和 volumes（数据清空！）
docker compose down -v

# 查看状态
docker compose ps

# 查看日志
docker compose logs -f
docker compose logs -f mlflow    # 指定服务

# 重新构建（代码改变后）
docker compose build jupyter
docker compose up -d --build jupyter

# 进入容器
docker compose exec jupyter bash

# 扩展服务实例（横向扩展）
docker compose up -d --scale inference=3
```

---

## 5. 环境变量管理

```bash
# .env 文件（不要提交到 git）
POSTGRES_PASSWORD=your-strong-password
MINIO_ROOT_PASSWORD=your-strong-password
MLFLOW_PORT=5000
```

```yaml
# docker-compose.yml 中引用
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
```

---

## 6. GPU 配置（Compose）

```yaml
services:
  training:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all         # 或 1, 2 等
              capabilities: [gpu]
```

---

## 练习

**Exercise 1:** 部署完整的 MLflow 平台，运行一个 sklearn 实验并在 UI 查看结果  
**Exercise 2:** 添加 Grafana + Prometheus 监控服务，监控推理服务的延迟指标  
**Exercise 3:** 实现训练服务和推理服务的 Compose 编排，训练完成后自动更新模型  
**Exercise 4:** 配置 GPU 训练服务，在 Compose 中同时运行 Jupyter 和训练任务
