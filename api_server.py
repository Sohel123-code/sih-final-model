"""
api_server.py — FastAPI REST API for CNN-Swin SAR Oil Thickness Classification

Model: cnn_swin_thickness_best.pth
Architecture: CNN (ResNet-18) + Swin Transformer (Swin-Tiny) -> 3 Thickness Classes

Endpoints:
  GET  /               — Health & model architecture info
  GET  /health         — Simple health check
  POST /predict        — Single image inference
  POST /predict-batch  — Multiple images batch inference
"""

import os
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import torch.nn.functional as F
import torchvision.transforms as T

from model import load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CKPT_THICKNESS = os.path.join(_BASE_DIR, "cnn_swin_thickness_best.pth")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_LABELS    = ["Thin_Sheen", "Moderate", "Thick_Emulsified"]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model Singleton
# ---------------------------------------------------------------------------
_thickness_model = None

def get_model():
    global _thickness_model
    if _thickness_model is None:
        print(f"[API] Loading CNN-Swin Thickness Classifier on {DEVICE}...")
        _thickness_model = load_model(CKPT_THICKNESS, device=DEVICE)
        print("[API] Model loaded successfully ✓")
    return _thickness_model

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SAR Oil Thickness Classification API (CNN-Swin)",
    description="Hybrid CNN (ResNet-18) + Swin Transformer (Swin-Tiny) inference for SAR oil spill thickness classification.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    filename: Optional[str]
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    architecture: str
    device: str

# ---------------------------------------------------------------------------
# Core Inference
# ---------------------------------------------------------------------------
def _run_inference(pil_img: Image.Image) -> dict:
    model = get_model()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()

    top_idx = int(np.argmax(probs))
    pred_label = CLASS_LABELS[top_idx]
    confidence = float(probs[top_idx])

    prob_dict = {label: float(round(probs[i], 4)) for i, label in enumerate(CLASS_LABELS)}

    return {
        "predicted_class": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "architecture": "ResNet-18 (512-d) + Swin-Tiny (768-d) -> Linear(512) -> Linear(3)",
        "device": DEVICE,
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "running",
        "model": "cnn_swin_thickness_best.pth",
        "classes": CLASS_LABELS,
        "backbone": "ResNet-18 + Swin-Tiny Hybrid",
        "device": DEVICE,
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict", response_model=PredictResponse)
async def predict_single(
    file: UploadFile = File(...),
):
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    result = _run_inference(pil_img)
    result["filename"] = file.filename
    return result

@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
):
    results = []

    for f in files:
        try:
            contents = await f.read()
            pil_img  = Image.open(io.BytesIO(contents))
            result   = _run_inference(pil_img)
            result["filename"] = f.filename
            result["error"]    = None
        except Exception as e:
            result = {
                "filename": f.filename,
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {},
                "architecture": "",
                "device": DEVICE,
                "error": str(e),
            }

        results.append(result)

    return {
        "total": len(files),
        "results": results,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
