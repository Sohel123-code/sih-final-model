"""
app.py -- Gradio UI for CNN-Swin SAR Oil Spill Detector

Model: cnn_swin_v2_best.pth
Architecture:
  - CNN Branch: ResNet-18 backbone (512-dim features)
  - Swin Branch: Swin-Tiny backbone (768-dim features)
  - Fusion Trunk: Concat (1280-dim) -> Linear(512) -> BN -> ReLU -> Dropout(0.3)
                  -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(2)
  - Output Classes: [0: No_Oil, 1: Oil]
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import glob
import torch
import numpy as np
from PIL import Image
import gradio as gr
from model import load_oil_detector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CKPT_PATH = os.path.join(_BASE_DIR, "cnn_swin_v2_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

LABELS = ["No_Oil", "Oil"]
OIL_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Model Loader (Singleton)
# ---------------------------------------------------------------------------
_model = None

def get_model():
    """Lazy-load the CNN-Swin v2 Oil Detector model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_CKPT_PATH):
            raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_CKPT_PATH}")
        _model = load_oil_detector(MODEL_CKPT_PATH, device=DEVICE)
    return _model

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
def predict(image: Image.Image):
    if image is None:
        return "Please upload a Sentinel-1 SAR image first."
    try:
        tensor = preprocess(image)
        model = get_model()

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        p_no_oil = float(probs[0]) * 100
        p_oil    = float(probs[1]) * 100
        is_oil   = p_oil >= (OIL_THRESHOLD * 100)

        if is_oil:
            banner = (
                f'<div style="background:#FFEBEE; border-left:6px solid #B71C1C; '
                f'padding:16px 20px; border-radius:10px; margin-bottom:18px;">'
                f'<span style="font-size:1.5rem; font-weight:800; color:#B71C1C; '
                f'letter-spacing:0.5px;">⚠️ OIL SPILL DETECTED</span><br>'
                f'<span style="font-size:0.9rem; color:#444;">Sentinel-1 SAR feature backscatter suppression verified</span>'
                f'</div>'
            )
            oil_status = "🚨 High Confidence Spill" if p_oil > 80 else "⚠️ Possible Slick / Spill"
        else:
            banner = (
                f'<div style="background:#E8F5E9; border-left:6px solid #2E7D32; '
                f'padding:16px 20px; border-radius:10px; margin-bottom:18px;">'
                f'<span style="font-size:1.5rem; font-weight:800; color:#2E7D32; '
                f'letter-spacing:0.5px;">✅ NO OIL DETECTED</span><br>'
                f'<span style="font-size:0.9rem; color:#444;">Clean sea surface or standard oceanic signature</span>'
                f'</div>'
            )
            oil_status = "🛡️ Clean Sea"

        result = (
            f"{banner}\n\n"
            f"### 📊 Detection Analysis\n\n"
            f"| Metric | Value | Status |\n"
            f"|---|---|---|\n"
            f"| **Oil Spill Probability** | **`{p_oil:.2f}%`** | {'🔴 DETECTED' if is_oil else '⚪'} |\n"
            f"| **Clean Sea (No Oil)** | **`{p_no_oil:.2f}%`** | {'🟢 CLEAN' if not is_oil else '⚪'} |\n"
            f"| **Decision Status** | `{oil_status}` | Threshold: `{OIL_THRESHOLD*100:.0f}%` |\n\n"
            f"---\n\n"
            f"**🧠 Architecture Details**\n"
            f"- **Checkpoint**: `cnn_swin_v2_best.pth`\n"
            f"- **Backbone**: ResNet-18 (512-d) + Swin-Tiny (768-d)\n"
            f"- **Inference Device**: `{DEVICE.upper()}`\n"
        )
        return result

    except Exception as e:
        return f"**Error during inference:**\n\n```\n{e}\n```"

# ---------------------------------------------------------------------------
# Batch Inference (Multiple Images)
# ---------------------------------------------------------------------------
def predict_batch(images: list):
    if not images:
        return "Please upload one or more Sentinel-1 SAR images."

    model = get_model()
    results_parts = []
    oil_count = 0
    clean_count = 0

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

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            p_no_oil = float(probs[0]) * 100
            p_oil    = float(probs[1]) * 100
            is_oil   = p_oil >= (OIL_THRESHOLD * 100)

            if is_oil:
                oil_count += 1
                part = (
                    f'<div style="background:#FFEBEE; border-left:5px solid #B71C1C; '
                    f'padding:12px 16px; border-radius:8px; margin-bottom:10px;">'
                    f'<span style="font-size:1.15rem; font-weight:800; color:#B71C1C;">'
                    f'🚨 Image {idx}: {filename} &mdash; OIL DETECTED</span><br>'
                    f'<span style="font-size:0.85rem; color:#333;">Oil Probability: <b>{p_oil:.2f}%</b> &nbsp;|&nbsp; Clean Sea: {p_no_oil:.2f}%</span>'
                    f'</div>'
                )
            else:
                clean_count += 1
                part = (
                    f'<div style="background:#E8F5E9; border-left:5px solid #2E7D32; '
                    f'padding:12px 16px; border-radius:8px; margin-bottom:10px;">'
                    f'<span style="font-size:1.15rem; font-weight:800; color:#2E7D32;">'
                    f'✅ Image {idx}: {filename} &mdash; NO OIL</span><br>'
                    f'<span style="font-size:0.85rem; color:#333;">Clean Sea: <b>{p_no_oil:.2f}%</b> &nbsp;|&nbsp; Oil Probability: {p_oil:.2f}%</span>'
                    f'</div>'
                )

            results_parts.append(part)

        except Exception as e:
            results_parts.append(
                f"### Image {idx}\n\n**Error:** `{e}`\n\n"
            )

    header = (
        f'<div style="background:#E3F2FD; border-left:6px solid #1565C0; '
        f'padding:14px 18px; border-radius:8px; margin-bottom:16px;">'
        f'<span style="font-size:1.3rem; font-weight:800; color:#0D47A1;">'
        f'Batch Analysis Complete &mdash; {len(images)} image(s) processed</span><br>'
        f'<span style="font-size:0.92rem; color:#333;">'
        f'🚨 Oil Detected: <b>{oil_count}</b> &nbsp;|&nbsp; ✅ Clean: <b>{clean_count}</b></span>'
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
            <h1>🛰️ Sentinel-1 SAR Oil Spill Detector</h1>
            <p>Hybrid CNN (ResNet-18) + Swin Transformer (Swin-Tiny) Deep Learning Architecture &nbsp;&middot;&nbsp;
               <span>cnn_swin_v2_best.pth</span>
            </p>
        </div>
        """)

        # Architecture Explainer
        with gr.Accordion("ℹ️ Model Architecture & Pipeline Details", open=False):
            gr.Markdown("""
### 🧠 CNN-Swin Hybrid Architecture (`cnn_swin_v2_best.pth`)

| Branch | Backbone | Features | Purpose |
|---|---|---|---|
| **CNN Branch** | **ResNet-18** (4 stages) | 512-dim | Captures local spatial textures, sharp edges, and slick boundaries |
| **Swin Branch** | **Swin-Tiny** (Window 7, Patch 4) | 768-dim | Captures global contextual dependencies across the SAR scene |
| **Feature Fusion** | Concatenation | **1280-dim** | Combined representation |
| **Classification Head** | Deep MLP Trunk | 2 classes | `1280 → Linear(512) → BN → ReLU → Dropout(0.3) → Linear(128) → ReLU → Dropout(0.3) → Linear(2)` |

- **Classes**: `0: No_Oil` (Clean Sea / Look-alikes), `1: Oil` (Marine Oil Spill)
- **Input Dimensions**: Sentinel-1 SAR normalized tensor `(3, 224, 224)`
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
                                "🔍 Detect Oil Spill",
                                elem_id="predict-btn",
                                variant="primary",
                            )
                            clear_btn = gr.ClearButton(
                                [image_input],
                                value="Clear",
                                variant="secondary",
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### Detection Result")
                        output_md = gr.Markdown(
                            value="*Upload a Sentinel-1 SAR image and click **Detect Oil Spill** to analyze.*",
                            elem_classes=["output-md"],
                        )
                        gr.HTML(f"""
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            <span class="badge">Device: {device_label}</span>
                            <span class="badge">Model: cnn_swin_v2_best.pth</span>
                            <span class="badge">Backbone: ResNet-18 + Swin-Tiny</span>
                        </div>
                        """)

                # Sample images if present
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
                gr.Markdown("### Upload Multiple SAR Images\nRun high-throughput oil spill detection on multiple Sentinel-1 SAR scenes simultaneously.")
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
                            <span class="badge">Mode: High-Throughput Batch</span>
                        </div>
                        """)

                batch_btn.click(fn=predict_batch, inputs=[batch_gallery], outputs=[batch_output_md], api_name="predict_batch")

        # Footer
        gr.HTML("""
        <p style="text-align:center; font-size:0.85rem; color:#777; margin-top:24px;">
            Sentinel-1 SAR Oil Spill Detector &nbsp;&middot;&nbsp; CNN + Swin Transformer Hybrid Architecture
        </p>
        """)

    return demo

# ---------------------------------------------------------------------------
# Module level initialization (Hugging Face Spaces compatible)
# ---------------------------------------------------------------------------
print(f"[Init] PyTorch: {torch.__version__} | Device: {DEVICE}")
print(f"[Init] Loading CNN-Swin v2 model ({MODEL_CKPT_PATH})...")
get_model()
print("[Init] CNN-Swin v2 model loaded successfully ✓")

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
