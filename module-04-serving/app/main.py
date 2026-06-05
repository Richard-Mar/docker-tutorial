from pydantic import BaseModel
from .model import predict
from fastapi import FastAPI, Request


app = FastAPI()

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/debug")
def debug(request: Request):
    return {"path": request.url.path}

@app.api_route("/{full_path:path}", methods=["GET", "POST"])
def catch_all(full_path: str, request: Request):
    return {"received_path": request.url.path}
@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    return predict(request.features)
