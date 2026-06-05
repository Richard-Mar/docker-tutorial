# Module 06: MLOps 实战

## 学习目标

- 用 Docker 实现可复现的 ML 实验
- 构建 CI/CD 自动构建和推送镜像流水线
- 镜像版本管理和发布策略
- 生产环境最佳实践

---

## 1. 可复现实验的黄金标准

```dockerfile
# 可复现性三要素：固定基础镜像 + 固定依赖 + 固定代码
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime   # 精确 tag，不用 latest

WORKDIR /workspace

# 完全 pin 住版本
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# requirements.txt 内容：
# torch==2.2.0  (已由基础镜像提供)
# transformers==4.37.0
# datasets==2.17.0
# accelerate==0.27.2
# wandb==0.16.3

COPY . .

# 把 git commit 写入镜像（追溯代码版本）
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

CMD ["python", "train.py"]
```

```bash
# 构建时注入 git commit
docker build \
  --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
  -t my-train:$(git rev-parse --short HEAD) .
```

---

## 2. 镜像版本管理策略

### 2.1 打标签（Tagging）

```bash
# 用 git commit SHA 作为镜像 tag（精确追溯）
docker build -t registry/my-model:abc1234 .

# 同时打 latest（方便引用）
docker tag registry/my-model:abc1234 registry/my-model:latest

# 语义化版本
docker tag registry/my-model:abc1234 registry/my-model:v1.2.0
```

### 2.2 推荐的命名规范

```
registry.example.com/{project}/{service}:{version}

例如：
  registry.cn-hangzhou.aliyuncs.com/myteam/text-classifier:v1.0.0
  ghcr.io/myorg/pytorch-trainer:2024.01-abc1234
```

---

## 3. GitHub Actions CI/CD 流水线

### 3.1 自动构建和推送镜像

```yaml
# .github/workflows/docker-build.yml
name: Build and Push Docker Image

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ml-trainer

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha,prefix=commit-

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          build-args: |
            GIT_COMMIT=${{ github.sha }}
          cache-from: type=gha        # 利用 GitHub Actions 缓存加速构建
          cache-to: type=gha,mode=max
```

### 3.2 CI 中跑训练测试

```yaml
  test-training:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build image
        run: docker build -t test-trainer .
      
      - name: Run smoke test（小数据集快速验证）
        run: |
          docker run --rm \
            -v $(pwd)/tests/data:/workspace/data \
            test-trainer \
            python train.py --max-steps 10 --smoke-test
```

---

## 4. 生产环境最佳实践

### 4.1 安全性

```dockerfile
# ❌ 不要以 root 运行
# ✓ 创建非 root 用户
RUN groupadd -r mluser && useradd -r -g mluser mluser
RUN chown -R mluser:mluser /workspace
USER mluser
```

```dockerfile
# ❌ 不要在镜像中写死敏感信息
ENV AWS_SECRET_KEY=xxxxx          # 危险！

# ✓ 通过运行时环境变量或 secrets 管理
# docker run -e AWS_SECRET_KEY=$AWS_SECRET_KEY ...
# 或使用 Docker Secrets（Swarm）/ Kubernetes Secrets
```

### 4.2 镜像安全扫描

```bash
# 用 Docker Scout 扫描漏洞
docker scout cves my-model:latest

# 用 Trivy 扫描
trivy image my-model:latest
```

### 4.3 LABEL 追踪元信息

```dockerfile
LABEL org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="2024-01-15" \
      maintainer="your-team@example.com" \
      model.framework="pytorch" \
      model.task="text-classification"
```

### 4.4 资源限制

```bash
# 限制容器资源使用（防止 OOM 影响其他服务）
docker run \
  --memory="16g" \
  --memory-swap="16g" \      # 禁用 swap
  --cpus="4" \
  my-inference:latest
```

---

## 5. 综合项目：端到端 ML 流水线

### 5.1 流水线架构

```
代码提交 (git push)
    ↓
GitHub Actions
    ↓
1. 构建训练镜像
2. 运行烟雾测试
3. 推送镜像到 Registry
    ↓
4. 触发训练任务（在 GPU 服务器）
    ↓
5. 训练完成 → 模型注册到 MLflow
    ↓
6. 构建推理镜像（含新模型权重）
7. 推送推理镜像
    ↓
8. 部署推理服务（更新 docker-compose）
```

### 5.2 完整 docker-compose.yml（生产版）

```yaml
version: "3.8"

services:
  # 推理服务（生产）
  inference:
    image: registry.example.com/myteam/text-classifier:${MODEL_VERSION}
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 8g
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/workspace/models/current
      - LOG_LEVEL=info
    volumes:
      - model-store:/workspace/models:ro   # 只读挂载
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      inference:
        condition: service_healthy

volumes:
  model-store:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /shared/models
```

---

## 6. 调试技巧

```bash
# 查看容器内的环境
docker run --rm my-train:latest env | sort

# 检查构建缓存使用情况
docker buildx du

# 查看镜像历史（每层大小）
docker history my-train:latest

# 导出镜像用于离线部署
docker save my-train:latest | gzip > my-train-latest.tar.gz
docker load < my-train-latest.tar.gz

# 进入已退出的容器（调试启动失败）
docker run -it --entrypoint bash my-train:latest
```

---

## 综合项目

实现一个完整的文本分类 MLOps 流水线：

1. **训练服务**：Dockerfile.train + GPU 支持 + MLflow 记录
2. **推理服务**：Dockerfile.inference + FastAPI + 健康检查
3. **实验平台**：docker-compose.yml（MLflow + MinIO + Jupyter）
4. **CI/CD**：GitHub Actions 自动构建推送
5. **文档**：镜像版本变更记录

---

## 学习完成后你能做什么

- 为任何 ML 项目编写 Dockerfile，保证环境一致性
- 搭建本地 MLflow + Jupyter 开发平台
- 用 FastAPI + Docker 部署推理服务
- 配置 GPU 容器进行分布式训练
- 建立 CI/CD 流水线自动化镜像发布
- 在云平台（AWS/GCP/阿里云）部署容器化 ML 服务
