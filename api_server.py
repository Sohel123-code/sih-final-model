"""
api_server.py — FastAPI REST API for CNN-Swin SAR Oil Spill Detection

Model: cnn_swin_v2_best.pth
Architecture: CNN (ResNet-18) + Swin Transformer (Swin-Tiny)

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
from typing import Optional
import torch.nn.functional as F
import torchvision.transforms as T

from model import load_oil_detector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CKPT_OIL_DETECTOR = os.path.join(_BASE_DIR, "cnn_swin_v2_best.pth")
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model Singleton
# ---------------------------------------------------------------------------
_oil_model = None

def get_model():
    global _oil_model
    if _oil_model is None:
        print(f"[API] Loading CNN-Swin v2 Oil Detector on {DEVICE}...")
        _oil_model = load_oil_detector(CKPT_OIL_DETECTOR, device=DEVICE)
        print("[API] Model loaded successfully ✓")
    return _oil_model

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SAR Oil Spill Detection API (CNN-Swin v2)",
    description="Hybrid CNN (ResNet-18) + Swin Transformer (Swin-Tiny) inference for Sentinel-1 SAR oil spill detection.",
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
    oil_probability: float
    no_oil_probability: float
    status: str
    architecture: str
    device: str

# ---------------------------------------------------------------------------
# Core Inference
# ---------------------------------------------------------------------------
def _run_inference(pil_img: Image.Image, oil_threshold: float = 0.5) -> dict:
    model = get_model()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()

    p_no_oil = float(probs[0])
    p_oil    = float(probs[1])
    is_oil   = p_oil >= oil_threshold

    return {
        "is_oil": is_oil,
        "oil_probability": round(p_oil, 4),
        "no_oil_probability": round(p_no_oil, 4),
        "status": "OIL_DETECTED" if is_oil else "CLEAN_SEA",
        "architecture": "ResNet-18 (512-d) + Swin-Tiny (768-d) -> Linear(2)",
        "device": DEVICE,
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "running",
        "model": "cnn_swin_v2_best.pth",
        "backbone": "ResNet-18 + Swin-Tiny Hybrid",
        "device": DEVICE,
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict", response_model=PredictResponse)
async def predict_single(
    file: UploadFile = File(...),
    oil_threshold: float = 0.5,
):
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    result = _run_inference(pil_img, oil_threshold=oil_threshold)
    result["filename"] = file.filename
    return result

@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    oil_threshold: float = 0.5,
):
    results = []
    oil_count = 0
    clean_count = 0

    for f in files:
        try:
            contents = await f.read()
            pil_img  = Image.open(io.BytesIO(contents))
            result   = _run_inference(pil_img, oil_threshold=oil_threshold)
            result["filename"] = f.filename
            result["error"]    = None

            if result["is_oil"]:
                oil_count += 1
            else:
                clean_count += 1
        except Exception as e:
            result = {
                "filename": f.filename,
                "is_oil": None,
                "error": str(e),
            }

        results.append(result)

    return {
        "total": len(files),
        "oil_detected": oil_count,
        "clean": clean_count,
        "results": results,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
