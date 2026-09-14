"""
app.py -- Gradio UI for CNN-Swin SAR Oil Thickness Classifier

Model: cnn_swin_thickness_best.pth
Architecture:
  - CNN Branch:  ResNet-18 backbone → 512-dim features
  - Swin Branch: Swin-Tiny backbone → 768-dim features
  - Fusion:      Concat(1280) → Linear(512) → BN → ReLU
  - Head:        Linear(512, 3) → [Thin_Sheen, Moderate, Thick_Emulsified]
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
from model import load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CKPT_PATH = os.path.join(_BASE_DIR, "cnn_swin_thickness_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

CLASS_LABELS = ["Thin_Sheen", "Moderate", "Thick_Emulsified"]
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
_model = None

def get_model():
    """Lazy-load the CNN-Swin thickness classifier."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_CKPT_PATH):
            raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_CKPT_PATH}")
        _model = load_model(MODEL_CKPT_PATH, device=DEVICE)
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

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_LABELS[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        pred_desc  = CLASS_DESCS[pred_class]
        bg_color, text_color, emoji = CLASS_COLORS[pred_class]

        # Build probability table rows
        rows = ""
        for i, cls in enumerate(CLASS_LABELS):
            prob = f"{probs[i]*100:.2f}%"
            _, _, cls_emoji = CLASS_COLORS[cls]
            if i == pred_idx:
                rows += f"| **{cls_emoji} {cls}** | **{prob}** | ✅ **Predicted** |\n"
            else:
                rows += f"| {cls_emoji} {cls} | {prob} | |\n"

        result = (
            f'<div style="background:{bg_color}; border-left:6px solid {text_color}; '
            f'padding:16px 20px; border-radius:10px; margin-bottom:18px;">'
            f'<span style="font-size:1.5rem; font-weight:800; color:{text_color}; '
            f'letter-spacing:0.5px;">{emoji} {pred_class.replace("_", " ").upper()}</span><br>'
            f'<span style="font-size:0.9rem; color:#444;">SAR Oil Thickness Classification Complete</span>'
            f'</div>\n\n'
            f"**Confidence:** `{confidence:.1f}%`\n\n"
            f"> {pred_desc}\n\n"
            f"---\n\n"
            f"### 📊 Class Probabilities\n\n"
            f"| Class | Probability | Status |\n"
            f"|---|---|---|\n"
            f"{rows}\n"
            f"---\n\n"
            f"**🧠 Architecture**: ResNet-18 (512-d) + Swin-Tiny (768-d) → Fusion(1280→512) → Head(3)\n\n"
            f"**Checkpoint**: `cnn_swin_thickness_best.pth` &nbsp;|&nbsp; **Device**: `{DEVICE.upper()}`\n"
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
    class_counts = {cls: 0 for cls in CLASS_LABELS}

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

            pred_idx   = int(np.argmax(probs))
            pred_class = CLASS_LABELS[pred_idx]
            confidence = float(probs[pred_idx]) * 100
            bg_color, text_color, emoji = CLASS_COLORS[pred_class]
            class_counts[pred_class] += 1

            # Per-class probabilities inline
            prob_str = " &nbsp;|&nbsp; ".join(
                [f"{CLASS_COLORS[cls][2]} {cls}: {probs[i]*100:.1f}%" for i, cls in enumerate(CLASS_LABELS)]
            )

            part = (
                f'<div style="background:{bg_color}; border-left:5px solid {text_color}; '
                f'padding:12px 16px; border-radius:8px; margin-bottom:10px;">'
                f'<span style="font-size:1.15rem; font-weight:800; color:{text_color};">'
                f'{emoji} Image {idx}: {filename} &mdash; {pred_class}</span><br>'
                f'<span style="font-size:0.85rem; color:#333;">Confidence: <b>{confidence:.1f}%</b> &nbsp;|&nbsp; {prob_str}</span>'
                f'</div>'
            )
            results_parts.append(part)

        except Exception as e:
            results_parts.append(f"### Image {idx}\n\n**Error:** `{e}`\n\n")

    # Summary counts
    counts_str = " &nbsp;|&nbsp; ".join(
        [f"{CLASS_COLORS[cls][2]} {cls}: **{class_counts[cls]}**" for cls in CLASS_LABELS]
    )

    header = (
        f'<div style="background:#E3F2FD; border-left:6px solid #1565C0; '
        f'padding:14px 18px; border-radius:8px; margin-bottom:16px;">'
        f'<span style="font-size:1.3rem; font-weight:800; color:#0D47A1;">'
        f'Batch Analysis Complete &mdash; {len(images)} image(s) processed</span><br>'
        f'<span style="font-size:0.92rem; color:#333;">{counts_str}</span>'
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
            <h1>🛰️ SAR Oil Spill Thickness Classifier</h1>
            <p>Hybrid CNN (ResNet-18) + Swin Transformer (Swin-Tiny) &nbsp;&middot;&nbsp;
               <span>cnn_swin_thickness_best.pth &nbsp;&middot;&nbsp; 3-Class Thickness</span>
            </p>
        </div>
        """)

        # Architecture Explainer
        with gr.Accordion("ℹ️ Model Architecture & Thickness Classes", open=False):
            gr.Markdown("""
### 🧠 CNN-Swin Hybrid Architecture (`cnn_swin_thickness_best.pth`)

| Branch | Backbone | Features | Purpose |
|---|---|---|---|
| **CNN Branch** | **ResNet-18** (4 stages) | 512-dim | Local spatial textures, slick edges & boundaries |
| **Swin Branch** | **Swin-Tiny** (Window 7, Patch 4) | 768-dim | Global contextual dependencies across SAR scene |
| **Fusion Trunk** | `Linear(1280→512) → BN → ReLU` | 512-dim | Fused multi-scale representation |
| **Thickness Head** | `Linear(512→3)` | 3 classes | Oil thickness prediction |

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
                                "🔍 Classify Thickness",
                                elem_id="predict-btn",
                                variant="primary",
                            )
                            clear_btn = gr.ClearButton(
                                [image_input],
                                value="Clear",
                                variant="secondary",
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### Classification Result")
                        output_md = gr.Markdown(
                            value="*Upload a Sentinel-1 SAR image and click **Classify Thickness** to analyze.*",
                            elem_classes=["output-md"],
                        )
                        gr.HTML(f"""
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            <span class="badge">Device: {device_label}</span>
                            <span class="badge">Model: cnn_swin_thickness_best.pth</span>
                            <span class="badge">Classes: Thin / Moderate / Thick</span>
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
                gr.Markdown("### Upload Multiple SAR Images\nRun thickness classification on multiple Sentinel-1 SAR scenes simultaneously.")
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
                            batch_btn = gr.Button("⚡ Classify All Images", variant="primary")
                            batch_clear_btn = gr.ClearButton([batch_gallery], value="Clear All", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Summary & Results")
                        batch_output_md = gr.Markdown(
                            value="*Upload multiple SAR images and click **Classify All Images**.*",
                            elem_classes=["output-md"],
                        )
                        gr.HTML(f"""
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            <span class="badge">Device: {device_label}</span>
                            <span class="badge">Mode: Batch Thickness Classification</span>
                        </div>
                        """)

                batch_btn.click(fn=predict_batch, inputs=[batch_gallery], outputs=[batch_output_md], api_name="predict_batch")

        # Footer
        gr.HTML("""
        <p style="text-align:center; font-size:0.85rem; color:#777; margin-top:24px;">
            SAR Oil Thickness Classifier &nbsp;&middot;&nbsp; CNN + Swin Transformer Hybrid &nbsp;&middot;&nbsp; cnn_swin_thickness_best.pth
        </p>
        """)

    return demo

# ---------------------------------------------------------------------------
# Module level initialization (Hugging Face Spaces compatible)
# ---------------------------------------------------------------------------
print(f"[Init] PyTorch: {torch.__version__} | Device: {DEVICE}")
print(f"[Init] Loading CNN-Swin thickness model ({MODEL_CKPT_PATH})...")
get_model()
print("[Init] Thickness model loaded successfully ✓")

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
