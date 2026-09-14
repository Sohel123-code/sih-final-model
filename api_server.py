"""
api_server.py — FastAPI REST API for Hierarchical CNN-Swin SAR Oil Detection & Thickness Classifier

Stage 1: Binary Oil / Non-Oil Detection (cnn_swin_v2_best.pth)
Stage 2: 3-Class Oil Thickness Classifier (cnn_swin_thickness_best.pth)

Endpoints:
  GET  /               — Health & model info
  GET  /health         — Health check
  POST /predict        — Single image hierarchical inference
  POST /predict-batch  — Multiple images batch hierarchical inference
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

from model import load_oil_detector, load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CKPT_OIL_DETECTOR = os.path.join(_BASE_DIR, "cnn_swin_v2_best.pth")
CKPT_THICKNESS    = os.path.join(_BASE_DIR, "cnn_swin_thickness_best.pth")
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
THICKNESS_CLASSES = ["Thin_Sheen", "Moderate", "Thick_Emulsified"]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model Singleton
# ---------------------------------------------------------------------------
_oil_detector = None
_thickness_model = None

def get_models():
    global _oil_detector, _thickness_model
    if _oil_detector is None:
        print(f"[API] Loading Stage 1 Oil Detector on {DEVICE}...")
        _oil_detector = load_oil_detector(CKPT_OIL_DETECTOR, device=DEVICE)
    if _thickness_model is None:
        print(f"[API] Loading Stage 2 Thickness Model on {DEVICE}...")
        _thickness_model = load_model(CKPT_THICKNESS, device=DEVICE)
    return _oil_detector, _thickness_model

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SAR Oil Detection & Thickness Classification API (CNN-Swin)",
    description="Hierarchical CNN (ResNet-18) + Swin Transformer inference for Sentinel-1 SAR oil spill detection & thickness classification.",
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
    is_oil: bool
    status: str
    oil_probability: float
    no_oil_probability: float
    thickness_class: Optional[str] = None
    thickness_confidence: Optional[float] = None
    thickness_probabilities: Optional[Dict[str, float]] = None
    architecture: str
    device: str

# ---------------------------------------------------------------------------
# Core Inference
# ---------------------------------------------------------------------------
def _run_inference(pil_img: Image.Image) -> dict:
    oil_detector, thickness_model = get_models()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # Stage 1: Binary Detection
    with torch.no_grad():
        oil_logits = oil_detector(tensor)
        oil_probs  = F.softmax(oil_logits, dim=1).cpu().squeeze(0).numpy()

    p_no_oil = float(oil_probs[0])
    p_oil    = float(oil_probs[1])
    is_oil   = p_oil >= 0.5

    res = {
        "is_oil": is_oil,
        "status": "OIL_DETECTED" if is_oil else "CLEAN_SEA",
        "oil_probability": float(round(p_oil, 4)),
        "no_oil_probability": float(round(p_no_oil, 4)),
        "thickness_class": None,
        "thickness_confidence": None,
        "thickness_probabilities": None,
        "architecture": "Stage 1 (ResNet18+Swin-Tiny 2-class) -> Stage 2 (ResNet18+Swin-Tiny 3-class)",
        "device": DEVICE,
    }

    # Stage 2: Thickness Classification (Only if oil detected)
    if is_oil:
        with torch.no_grad():
            thick_logits = thickness_model(tensor)
            thick_probs  = F.softmax(thick_logits, dim=1).cpu().squeeze(0).numpy()

        top_idx = int(np.argmax(thick_probs))
        res["thickness_class"] = THICKNESS_CLASSES[top_idx]
        res["thickness_confidence"] = float(round(thick_probs[top_idx], 4))
        res["thickness_probabilities"] = {
            cls: float(round(thick_probs[i], 4)) for i, cls in enumerate(THICKNESS_CLASSES)
        }

    return res

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "running",
        "stage1_model": "cnn_swin_v2_best.pth",
        "stage2_model": "cnn_swin_thickness_best.pth",
        "thickness_classes": THICKNESS_CLASSES,
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
    oil_count = 0
    clean_count = 0
    thickness_counts = {cls: 0 for cls in THICKNESS_CLASSES}

    for f in files:
        try:
            contents = await f.read()
            pil_img  = Image.open(io.BytesIO(contents))
            result   = _run_inference(pil_img)
            result["filename"] = f.filename
            result["error"]    = None

            if result["is_oil"]:
                oil_count += 1
                if result["thickness_class"] in thickness_counts:
                    thickness_counts[result["thickness_class"]] += 1
            else:
                clean_count += 1
        except Exception as e:
            result = {
                "filename": f.filename,
                "is_oil": False,
                "status": "ERROR",
                "oil_probability": 0.0,
                "no_oil_probability": 0.0,
                "thickness_class": None,
                "thickness_confidence": None,
                "thickness_probabilities": None,
                "architecture": "",
                "device": DEVICE,
                "error": str(e),
            }

        results.append(result)

    return {
        "total": len(files),
        "oil_detected_count": oil_count,
        "clean_sea_count": clean_count,
        "thickness_counts": thickness_counts,
        "results": results,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
