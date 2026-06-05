# Module 04: 模型服务化（推理服务）

## 学习目标

- 用 FastAPI 封装模型推理接口
- 为推理服务构建轻量级 Docker 镜像
- 配置健康检查和优雅停机
- 理解推理服务的镜像优化策略

---

## 1. 推理服务 vs 训练镜像

| 维度 | 训练镜像 | 推理镜像 |
|------|---------|---------|
| 基础镜像 | pytorch devel / NGC | pytorch runtime / python-slim |
| 体积 | 可以大（10-20GB） | 尽量小（1-5GB） |
| CUDA | devel（含编译工具）| runtime（仅运行库）|
| 包 | 训练工具（wandb, apex）| 推理工具（fastapi, uvicorn）|
| 模型权重 | 挂载 volume | 内置或挂载 |

---

## 2. 最简 FastAPI 推理服务

### 2.1 项目结构

```
module-04-serving/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 应用
│   └── model.py         # 模型加载和推理逻辑
├── models/              # 模型权重（volume 挂载）
├── Dockerfile
└── requirements.txt
```

### 2.2 模型推理代码

```python
# app/model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model

def predict(text: str, model_path: str = "/workspace/models/my-classifier"):
    tokenizer, model = load_model(model_path)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    return {
        "label": int(probs.argmax()),
        "score": float(probs.max()),
    }
```

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import predict

app = FastAPI(title="Text Classification API", version="1.0.0")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: int
    score: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    result = predict(request.text)
    return result
```

### 2.3 推理服务 Dockerfile

```dockerfile
# Dockerfile
# 使用 runtime 镜像（不含编译工具，体积更小）
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 只安装推理必要的包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# 声明端口
EXPOSE 8000

# 健康检查（Docker 会定期检查容器是否正常）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 生产用 gunicorn + uvicorn worker
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
```

```
# requirements.txt（推理服务专用）
fastapi==0.110.0
uvicorn[standard]==0.28.0
transformers==4.37.0
torch==2.2.0
pydantic==2.6.0
```

### 2.4 构建和运行

```bash
# 构建
docker build -t text-classifier:v1.0 .

# 运行（挂载模型权重，暴露端口）
docker run --rm \
  --gpus '"device=0"' \
  -p 8000:8000 \
  -v $(pwd)/models:/workspace/models \
  text-classifier:v1.0

# 测试
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great product!"}'
```

---

## 3. 镜像体积优化

### 3.1 CPU-only 推理（最小化体积）

```dockerfile
# 如果不需要 GPU，使用 slim Python 镜像
FROM python:3.11-slim

WORKDIR /workspace

RUN pip install --no-cache-dir \
    fastapi uvicorn \
    "transformers[onnx]" \   # 用 ONNX 运行时替代 PyTorch
    onnxruntime              # CPU 推理约 100MB vs PyTorch 700MB+

COPY app/ ./app/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.2 多阶段构建（分离依赖安装）

```dockerfile
# 阶段 1：安装依赖
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir --prefix=/install \
    fastapi uvicorn transformers torch

# 阶段 2：最终镜像
FROM python:3.11-slim

COPY --from=builder /install /usr/local
COPY app/ /workspace/app/

WORKDIR /workspace
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 4. 模型权重处理策略

### 策略 A：volume 挂载（推荐，灵活）

```bash
# 适合：模型经常更新、权重文件大
docker run -v /shared/models:/workspace/models text-classifier:v1.0
```

### 策略 B：内置在镜像（版本固定）

```dockerfile
# 适合：模型版本固定、需要独立部署
COPY models/my-classifier/ /workspace/models/my-classifier/
# 注意：会使镜像非常大（几GB）
```

### 策略 C：启动时下载（from HuggingFace Hub）

```dockerfile
# 下载脚本
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
# 或通过环境变量在启动时控制
ENV MODEL_NAME=bert-base-uncased
```

---

## 5. 环境变量配置

```dockerfile
# 通过环境变量控制服务行为（不 hardcode）
ENV MODEL_PATH=/workspace/models/my-classifier \
    MAX_WORKERS=1 \
    LOG_LEVEL=info \
    DEVICE=auto
```

```bash
# 运行时覆盖
docker run -e MODEL_PATH=/models/v2 -e MAX_WORKERS=2 text-classifier:v1.0
```

---

## 6. 测试容器服务

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 推理请求
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this movie!"}'

# 3. API 文档（FastAPI 自动生成）
open http://localhost:8000/docs

# 4. 压力测试
pip install locust
# 编写 locustfile.py 进行并发测试
```

---

## 练习

**Exercise 1:** 将你自己的 PyTorch 模型封装为 FastAPI 服务并容器化  
**Exercise 2:** 对比 GPU 镜像 vs CPU 镜像的推理速度和体积  
**Exercise 3:** 实现 HEALTHCHECK，模拟服务故障，观察容器状态变化  
**Exercise 4:** 用 ONNX Runtime 替换 PyTorch 做 CPU 推理，对比延迟
