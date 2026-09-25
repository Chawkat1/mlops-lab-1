"""
FastAPI serving app for the food11 model registered in mlflow.

Usage:
    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/food11@champion"
IMAGE_SIZE = 128

# Alphabetical order, matching torchvision.datasets.ImageFolder's class-to-index mapping
# used during training.
CLASS_NAMES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

app = FastAPI(title="food11-api")
model = None


@app.on_event("startup")
def load_model() -> None:
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0).numpy()

    logits = model.predict(input_tensor)
    logits = torch.as_tensor(np.asarray(logits))
    probs = torch.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    return {
        "category": CLASS_NAMES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
    }
