"""
app.py -- Gradio UI for Hierarchical CNN-Swin SAR Oil Spill Detection & Thickness Classifier

Stage 1: Binary Oil / Non-Oil Detection (cnn_swin_v2_best.pth)
Stage 2: 3-Class Oil Thickness Classifier (cnn_swin_thickness_best.pth)
"""

try:
    import spaces
except Exception:
    class spaces:
        @staticmethod
        def GPU(func=None, *args, **kwargs):
            if callable(func):
                return func
            def decorator(fn):
                return fn
            return decorator

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import glob
import torch
import numpy as np
from PIL import Image
import gradio as gr
from model import load_oil_detector, load_model
from noaa_oils import select_openoil_type_grounded

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_OIL_DETECTOR = os.path.join(_BASE_DIR, "cnn_swin_v2_best.pth")
CKPT_THICKNESS    = os.path.join(_BASE_DIR, "cnn_swin_thickness_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

THICKNESS_CLASSES = ["Thin_Sheen", "Moderate", "Thick_Emulsified"]
CLASS_DESCS = {
    "Thin_Sheen":       "Thin oil sheen — light surface film with minimal SAR backscatter dampening. High rate of natural dispersion expected.",
    "Moderate":         "Moderate emulsion — intermediate thickness with notable SAR signature. Mechanical containment recommended.",
    "Thick_Emulsified": "Thick emulsified oil — heavy emulsion with strong SAR backscatter suppression. Immediate response required.",
}
CLASS_COLORS = {
    "Thin_Sheen":       ("#FFF3E0", "#E65100", "🟡"),
    "Moderate":         ("#FFF8E1", "#F57F17", "🟠"),
    "Thick_Emulsified": ("#FFEBEE", "#B71C1C", "🔴"),
}

# ---------------------------------------------------------------------------
# Model Loader (Singleton)
# ---------------------------------------------------------------------------
_oil_detector = None
_thickness_model = None

def get_models():
    """Lazy-load both Stage 1 Oil Detector and Stage 2 Thickness Classifier."""
    global _oil_detector, _thickness_model
    if _oil_detector is None:
        if not os.path.exists(CKPT_OIL_DETECTOR):
            raise FileNotFoundError(f"Oil Detector checkpoint not found: {CKPT_OIL_DETECTOR}")
        _oil_detector = load_oil_detector(CKPT_OIL_DETECTOR, device=DEVICE)
    if _thickness_model is None:
        if not os.path.exists(CKPT_THICKNESS):
            raise FileNotFoundError(f"Thickness Model checkpoint not found: {CKPT_THICKNESS}")
        _thickness_model = load_model(CKPT_THICKNESS, device=DEVICE)
    return _oil_detector, _thickness_model

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def preprocess(pil_image: Image.Image) -> torch.Tensor:
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(DEVICE)

# ---------------------------------------------------------------------------
# Single Image Inference
# ---------------------------------------------------------------------------
@spaces.GPU
def predict(image: Image.Image):
    if image is None:
        return "Please upload a Sentinel-1 SAR image first."
    try:
        tensor = preprocess(image)
        oil_detector, thickness_model = get_models()

        # Stage 1: Binary Oil Detection (Oil vs Non-Oil)
        with torch.no_grad():
            oil_logits = oil_detector(tensor)
            oil_probs = torch.softmax(oil_logits, dim=1).squeeze(0).cpu().numpy()

        p_no_oil = float(oil_probs[0])
        p_oil    = float(oil_probs[1])
        is_oil   = p_oil >= 0.5

        if is_oil:
            # Stage 2: Thickness Classification (Only for detected oil)
            with torch.no_grad():
                thick_logits = thickness_model(tensor)
                thick_probs = torch.softmax(thick_logits, dim=1).squeeze(0).cpu().numpy()

            pred_idx   = int(np.argmax(thick_probs))
            pred_class = THICKNESS_CLASSES[pred_idx]
            thick_conf = float(thick_probs[pred_idx]) * 100
            pred_desc  = CLASS_DESCS[pred_class]
            bg_color, text_color, emoji = CLASS_COLORS[pred_class]

            # NOAA ADIOS oil lookup
            noaa_info = select_openoil_type_grounded(pred_class)
            noaa_name = noaa_info.get("real_oil_name", "N/A")
            noaa_api  = noaa_info.get("api_gravity", "N/A")
            noaa_visc = noaa_info.get("viscosity_cSt", "N/A")

            # Probability table
            thick_rows = ""
            for i, cls in enumerate(THICKNESS_CLASSES):
                prob = f"{thick_probs[i]*100:.2f}%"
                _, _, cls_emoji = CLASS_COLORS[cls]
                if i == pred_idx:
                    thick_rows += f"| **{cls_emoji} {cls}** | **{prob}** | ✅ **Predicted** |\n"
                else:
                    thick_rows += f"| {cls_emoji} {cls} | {prob} | |\n"

            result = (
                f'<div style="background:#FFEBEE; border-left:6px solid #C62828; '
                f'padding:16px 20px; border-radius:10px; margin-bottom:18px;">'
                f'<span style="font-size:1.5rem; font-weight:800; color:#C62828; '
                f'letter-spacing:0.5px;">🛢️ OIL SPILL DETECTED</span><br>'
                f'<span style="font-size:0.95rem; color:#333;">Stage 1 Detection Confidence: <b>{p_oil*100:.1f}%</b></span>'
                f'</div>\n\n'
                f'<div style="background:{bg_color}; border-left:6px solid {text_color}; '
                f'padding:14px 18px; border-radius:8px; margin-bottom:16px;">'
                f'<span style="font-size:1.2rem; font-weight:800; color:{text_color};">'
                f'{emoji} Stage 2 Thickness Class: {pred_class.replace("_", " ").upper()} ({thick_conf:.1f}% confidence)</span>'
                f'</div>\n\n'
                f"> {pred_desc}\n\n"
                f"---\n\n"
                f"### 📊 Stage 2 Thickness Probabilities\n\n"
                f"| Thickness Class | Probability | Status |\n"
                f"|---|---|---|\n"
                f"{thick_rows}\n"
                f"---\n\n"
                f"### 🏛️ NOAA ADIOS Grounded Oil Properties\n\n"
                f"- **Mapped Oil Name**: `{noaa_name}`\n"
                f"- **API Gravity**: `{noaa_api}°`\n"
                f"- **Viscosity**: `{noaa_visc} cSt`\n\n"
                f"---\n\n"
                f"**🧠 Pipeline**: Stage 1 (`cnn_swin_v2_best.pth`) → Stage 2 (`cnn_swin_thickness_best.pth`)\n\n"
                f"**Device**: `{DEVICE.upper()}`\n"
            )
            return result
        else:
            # Clean Sea / Non-Oil
            result = (
                f'<div style="background:#E8F5E9; border-left:6px solid #2E7D32; '
                f'padding:16px 20px; border-radius:10px; margin-bottom:18px;">'
                f'<span style="font-size:1.5rem; font-weight:800; color:#2E7D32; '
                f'letter-spacing:0.5px;">🌊 CLEAN SEA (NON-OIL)</span><br>'
                f'<span style="font-size:0.95rem; color:#333;">Clean Sea Confidence: <b>{p_no_oil*100:.1f}%</b></span>'
                f'</div>\n\n'
                f"No oil spill detected in this Sentinel-1 SAR scene. Stage 2 thickness hazard evaluation skipped.\n\n"
                f"---\n\n"
                f"**Stage 1 Detector**: `cnn_swin_v2_best.pth` &nbsp;|&nbsp; **Device**: `{DEVICE.upper()}`\n"
            )
            return result

    except Exception as e:
        return f"**Error during inference:**\n\n```\n{e}\n```"

# ---------------------------------------------------------------------------
# Batch Inference (Multiple Images)
# ---------------------------------------------------------------------------
@spaces.GPU
def predict_batch(images: list):
    if not images:
        return "Please upload one or more Sentinel-1 SAR images."

    oil_detector, thickness_model = get_models()
    results_parts = []
    
    oil_count = 0
    clean_count = 0
    thickness_counts = {cls: 0 for cls in THICKNESS_CLASSES}

    for idx, img_data in enumerate(images, start=1):
        try:
            if isinstance(img_data, tuple):
                pil_img = img_data[0]
                filename = img_data[1] if img_data[1] else f"Image {idx}"
            elif isinstance(img_data, str):
                pil_img = Image.open(img_data)
                filename = os.path.basename(img_data)
            elif isinstance(img_data, Image.Image):
                pil_img = img_data
                filename = f"Image {idx}"
            else:
                pil_img = Image.open(img_data)
                filename = getattr(img_data, 'name', f"Image {idx}")

            if isinstance(filename, str) and (os.sep in filename or '/' in filename):
                filename = os.path.basename(filename)

            tensor = preprocess(pil_img)

            # Stage 1: Binary Detection
            with torch.no_grad():
                oil_logits = oil_detector(tensor)
                oil_probs = torch.softmax(oil_logits, dim=1).squeeze(0).cpu().numpy()

            p_no_oil = float(oil_probs[0])
            p_oil    = float(oil_probs[1])
            is_oil   = p_oil >= 0.5

            if is_oil:
                oil_count += 1
                # Stage 2: Thickness
                with torch.no_grad():
                    thick_logits = thickness_model(tensor)
                    thick_probs = torch.softmax(thick_logits, dim=1).squeeze(0).cpu().numpy()

                pred_idx   = int(np.argmax(thick_probs))
                pred_class = THICKNESS_CLASSES[pred_idx]
                confidence = float(thick_probs[pred_idx]) * 100
                bg_color, text_color, emoji = CLASS_COLORS[pred_class]
                thickness_counts[pred_class] += 1

                part = (
                    f'<div style="background:{bg_color}; border-left:5px solid {text_color}; '
                    f'padding:12px 16px; border-radius:8px; margin-bottom:10px;">'
                    f'<span style="font-size:1.15rem; font-weight:800; color:{text_color};">'
                    f'🛢️ Image {idx}: {filename} &mdash; OIL DETECTED ({pred_class})</span><br>'
                    f'<span style="font-size:0.85rem; color:#333;">Oil Prob: <b>{p_oil*100:.1f}%</b> &nbsp;|&nbsp; '
                    f'Thickness Conf: <b>{confidence:.1f}%</b> ({emoji} {pred_class})</span>'
                    f'</div>'
                )
            else:
                clean_count += 1
                part = (
                    f'<div style="background:#E8F5E9; border-left:5px solid #2E7D32; '
                    f'padding:12px 16px; border-radius:8px; margin-bottom:10px;">'
                    f'<span style="font-size:1.15rem; font-weight:800; color:#2E7D32;">'
                    f'🌊 Image {idx}: {filename} &mdash; CLEAN SEA (NON-OIL)</span><br>'
                    f'<span style="font-size:0.85rem; color:#333;">Clean Sea Confidence: <b>{p_no_oil*100:.1f}%</b></span>'
                    f'</div>'
                )

            results_parts.append(part)

        except Exception as e:
            results_parts.append(f"### Image {idx}\n\n**Error:** `{e}`\n\n")

    # Summary counts displayed prominently in the UI
    thick_breakdown = " &nbsp;|&nbsp; ".join(
        [f"{CLASS_COLORS[cls][2]} {cls}: <b>{thickness_counts[cls]}</b>" for cls in THICKNESS_CLASSES]
    )

    header = (
        f'<div style="background:#ECEFF1; border-left:6px solid #37474F; '
        f'padding:16px 20px; border-radius:10px; margin-bottom:18px;">'
        f'<span style="font-size:1.35rem; font-weight:800; color:#263238;">'
        f'📊 Batch Analysis Summary &mdash; {len(images)} Scenes Processed</span><br><br>'
        f'<div style="display:flex; gap:16px; flex-wrap:wrap; font-size:1.05rem;">'
        f'<span style="background:#E8F5E9; color:#1B5E20; padding:6px 14px; border-radius:6px; border:1px solid #A5D6A7;">'
        f'🌊 <b>Non-Oils (Clean Sea): {clean_count}</b></span>'
        f'<span style="background:#FFEBEE; color:#B71C1C; padding:6px 14px; border-radius:6px; border:1px solid #EF9A9A;">'
        f'🛢️ <b>Oils Detected: {oil_count}</b></span>'
        f'</div>'
        f'<div style="margin-top:10px; font-size:0.9rem; color:#455A64;">'
        f'<b>Oil Thickness Breakdown:</b> {thick_breakdown}'
        f'</div>'
        f'</div>\n\n'
    )

    return header + "\n".join(results_parts) + f"\n\n*Device: {DEVICE.upper()}*"

# ---------------------------------------------------------------------------
# Theme & CSS
# ---------------------------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("Roboto Mono"),
)

CUSTOM_CSS = """
#header-banner {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%);
    color: #FFFFFF !important;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(27, 94, 32, 0.2);
}
#header-banner h1 {
    font-size: 2rem;
    font-weight: 900;
    margin: 0 0 6px 0;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
}
#header-banner p {
    font-size: 0.95rem;
    color: #C8E6C9 !important;
    margin: 0;
}
.badge {
    display: inline-block;
    background: #E8F5E9;
    color: #1B5E20;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #A5D6A7;
}
footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks() as demo:
        # Header
        gr.HTML("""
        <div id="header-banner">
            <h1>🛰️ Hierarchical SAR Oil Spill Detector & Thickness Classifier</h1>
            <p>Stage 1: Binary Oil Detection (cnn_swin_v2_best.pth) &nbsp;&middot;&nbsp;
               Stage 2: 3-Class Thickness Classification (cnn_swin_thickness_best.pth)
            </p>
        </div>
        """)

        # Architecture Explainer
        with gr.Accordion("ℹ️ Hierarchical Pipeline Architecture & Thickness Classes", open=False):
            gr.Markdown("""
### 🧠 Hierarchical Pipeline Architecture

```
                      Input SAR Image (224×224)
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   Stage 1: Oil Detector      │
                  │   (cnn_swin_v2_best.pth)     │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────┴───────────────┐
                  ▼                              ▼
            [CLEAN SEA (NON-OIL)]          [OIL DETECTED]
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  Stage 2: Thickness Model    │
                                  │(cnn_swin_thickness_best.pth) │
                                  └──────────────┬───────────────┘
                                                 │
                                  ┌──────────────┼──────────────┐
                                  ▼              ▼              ▼
                            [Thin_Sheen]    [Moderate]  [Thick_Emulsified]
```

### 🛢️ Thickness Classes

| Class | Description | SAR Signature |
|---|---|---|
| 🟡 **Thin_Sheen** | Light surface film, minimal impact | Faint backscatter dampening |
| 🟠 **Moderate** | Intermediate emulsion, notable extent | Clear SAR signature |
| 🔴 **Thick_Emulsified** | Heavy emulsion, immediate response needed | Strong backscatter suppression |

**Input**: Sentinel-1 SAR image normalized to `(3, 224, 224)`
            """)

        device_label = "GPU (CUDA)" if DEVICE == "cuda" else "CPU"

        with gr.Tabs():
            # Tab 1: Single Image
            with gr.TabItem("🖼️ Single Image Analysis"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload SAR Image")
                        image_input = gr.Image(
                            type="pil",
                            label="Sentinel-1 SAR Scene",
                            height=340,
                        )
                        with gr.Row():
                            predict_btn = gr.Button(
                                "🔍 Run Detection & Thickness Analysis",
                                elem_id="predict-btn",
                                variant="primary",
                            )
                            clear_btn = gr.ClearButton(
                                [image_input],
                                value="Clear",
                                variant="secondary",
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### Analysis Result")
                        output_md = gr.Markdown(
                            value="*Upload a Sentinel-1 SAR image and click **Run Detection & Thickness Analysis**.*",
                            elem_classes=["output-md"],
                        )
                        gr.HTML(f"""
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            <span class="badge">Device: {device_label}</span>
                            <span class="badge">Stage 1: cnn_swin_v2_best.pth</span>
                            <span class="badge">Stage 2: cnn_swin_thickness_best.pth</span>
                        </div>
                        """)

                # Sample images
                example_dir = os.path.dirname(os.path.abspath(__file__))
                example_images = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
                    example_images += glob.glob(os.path.join(example_dir, ext))
                if example_images:
                    gr.Examples(
                        examples=[[img] for img in example_images[:6]],
                        inputs=image_input,
                        label="Sample SAR Scenes",
                    )

                predict_btn.click(fn=predict, inputs=[image_input], outputs=[output_md], api_name="predict")
                image_input.upload(fn=predict, inputs=[image_input], outputs=[output_md])

            # Tab 2: Batch Analysis
            with gr.TabItem("📂 Batch SAR Processing"):
                gr.Markdown("### Upload Multiple SAR Images\nRun hierarchical detection and thickness classification on multiple Sentinel-1 SAR scenes simultaneously.")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        batch_gallery = gr.Gallery(
                            label="Upload Multiple Images",
                            type="pil",
                            columns=3,
                            height=360,
                            object_fit="contain",
                        )
                        with gr.Row():
                            batch_btn = gr.Button("⚡ Analyze All Images", variant="primary")
                            batch_clear_btn = gr.ClearButton([batch_gallery], value="Clear All", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Summary & Results")
                        batch_output_md = gr.Markdown(
                            value="*Upload multiple SAR images and click **Analyze All Images**.*",
                            elem_classes=["output-md"],
                        )
                        gr.HTML(f"""
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            <span class="badge">Device: {device_label}</span>
                            <span class="badge">Mode: Hierarchical Batch Pipeline</span>
                        </div>
                        """)

                batch_btn.click(fn=predict_batch, inputs=[batch_gallery], outputs=[batch_output_md], api_name="predict_batch")

        # Footer
        gr.HTML("""
        <p style="text-align:center; font-size:0.85rem; color:#777; margin-top:24px;">
            Hierarchical SAR Oil Spill Detector & Thickness Classifier &nbsp;&middot;&nbsp; CNN + Swin Transformer Hybrid
        </p>
        """)

    return demo

# ---------------------------------------------------------------------------
# Module level initialization (Hugging Face Spaces compatible)
# ---------------------------------------------------------------------------
print(f"[Init] PyTorch: {torch.__version__} | Device: {DEVICE}")
print("[Init] Loading Stage 1 (Oil Detector) and Stage 2 (Thickness Model)...")
get_models()
print("[Init] Both models loaded successfully ✓")

demo = build_app()

if __name__ == "__main__":
    print("Launching Gradio app on http://0.0.0.0:7860 ...\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=False,
        theme=custom_theme,
        css=CUSTOM_CSS,
    )
