"""
app.py -- Gradio UI for CNN-Swin SAR Oil Thickness Classifier

Two-stage hierarchical inference pipeline:
  Stage 1: Oil/No-Oil detection   (cnn_swin_v2_best.pth  → 2 classes)
  Stage 2: Thickness classification (cnn_swin_thickness_best.pth → 3 classes)
           Only executed when Stage 1 predicts Oil.

Theme: White background, dark green (#1B5E20) text
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
from model import load_model, load_oil_detector
from noaa_oils import select_openoil_type_grounded

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stage 1: Oil / No-Oil detector
OIL_CKPT_PATH = os.path.join(_BASE_DIR, "cnn_swin_v2_best.pth")

# Stage 2: Thickness classifier (only runs when oil is detected)
THICKNESS_CKPT_PATH = os.path.join(_BASE_DIR, "cnn_swin_thickness_best.pth")

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 224

# Stage 1 labels
OIL_LABELS = ["No_Oil", "Oil"]

# Stage 2 labels (only used after Stage 1 confirms oil)
CLASS_LABELS = ["Thin_Sheen", "Moderate", "Thick_Emulsified"]
CLASS_DESCS  = [
    "Thin oil sheen -- light surface film with minimal SAR backscatter dampening.",
    "Moderate emulsion -- intermediate thickness with notable SAR signature.",
    "Thick emulsified oil -- heavy emulsion with strong SAR backscatter suppression.",
]

# Oil detection threshold (Stage 1 oil probability must exceed this)
OIL_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Models (lazy singletons)
# ---------------------------------------------------------------------------
_oil_model = None       # Stage 1
_thickness_model = None  # Stage 2

def get_oil_model():
    """Lazy-load the Stage 1 Oil/No-Oil detector."""
    global _oil_model
    if _oil_model is None:
        _oil_model = load_oil_detector(OIL_CKPT_PATH, device=DEVICE)
    return _oil_model

def get_thickness_model():
    """Lazy-load the Stage 2 thickness classifier."""
    global _thickness_model
    if _thickness_model is None:
        _thickness_model = load_model(THICKNESS_CKPT_PATH, device=DEVICE)
    return _thickness_model

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
# Two-Stage Inference — single image
# ---------------------------------------------------------------------------
# Stage 1: Oil/No-Oil detection (binary classifier)
# Stage 2: Thickness classification (only if Stage 1 → Oil)
# ---------------------------------------------------------------------------
def predict(image: Image.Image):
    if image is None:
        return "Please upload a Sentinel-1 SAR image first."
    try:
        tensor = preprocess(image)

        # ── Stage 1: Oil / No-Oil detection ──
        oil_model = get_oil_model()
        with torch.no_grad():
            oil_logits = oil_model(tensor)
        oil_probs = torch.softmax(oil_logits, dim=1).squeeze(0).cpu().numpy()
        oil_prob  = float(oil_probs[1]) * 100   # probability of class 1 (Oil)
        is_oil    = oil_prob >= (OIL_THRESHOLD * 100)

        if not is_oil:
            # ── No Oil → stop here, do NOT run thickness model ──
            result = (
                f'<div style="background:#E8F5E9; border-left:5px solid #2E7D32; '
                f'padding:12px 16px; border-radius:8px; margin-bottom:16px;">'
                f'<span style="font-size:1.4rem; font-weight:800; color:#2E7D32; '
                f'letter-spacing:0.5px;">NO OIL DETECTED</span><br>'
                f'<span style="font-size:0.85rem; color:#333;">Sentinel-1 SAR analysis complete</span>'
                f'</div>\n\n'
                f"**Oil Probability:** `{oil_prob:.1f}%`\n\n"
                f"**Stage 2:** *Not executed* — oil probability below threshold.\n\n"
                f"*Device: {DEVICE.upper()}*"
            )
            return result

        # ── Stage 2: Thickness classification (oil confirmed) ──
        thickness_model = get_thickness_model()
        with torch.no_grad():
            thickness_logits = thickness_model(tensor)

        probs      = torch.softmax(thickness_logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_LABELS[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        pred_desc  = CLASS_DESCS[pred_idx]

        # Build per-class rows
        rows = ""
        for i in range(len(CLASS_LABELS)):
            cls  = CLASS_LABELS[i]
            prob = f"{probs[i]*100:.2f}%"
            if i == pred_idx:
                rows += f"| **{cls}** | **{prob}** | ✅ Predicted |\n"
            else:
                rows += f"| {cls} | {prob} | |\n"

        # ── NOAA ADIOS oil match ──
        noaa = select_openoil_type_grounded(pred_class)
        noaa_section = "\n---\n\n**🛢️ NOAA ADIOS Oil Match**\n\n"
        if noaa.get("real_oil_name"):
            noaa_section += (
                f"| Property | Value |\n"
                f"|---|---|\n"
                f"| **Oil Name** | {noaa['real_oil_name']} |\n"
                f"| **API Gravity** | {noaa.get('api_gravity', 'N/A')} |\n"
                f"| **Source Location** | {noaa.get('source_location', 'N/A')} |\n"
                f"| **NOAA Labels** | {', '.join(noaa.get('noaa_labels', []))} |\n"
                f"| **OpenDrift Oil Type** | `{noaa.get('opendrift_oiltype', 'N/A')}` |\n\n"
            )
        else:
            noaa_section += (
                f"Using generic placeholder: `{noaa.get('opendrift_oiltype', 'N/A')}`\n\n"
            )

        result = (
            f'<div style="background:#FFEBEE; border-left:5px solid #B71C1C; '
            f'padding:12px 16px; border-radius:8px; margin-bottom:16px;">'
            f'<span style="font-size:1.4rem; font-weight:800; color:#B71C1C; '
            f'letter-spacing:0.5px;">OIL DETECTED</span><br>'
            f'<span style="font-size:0.85rem; color:#333;">Sentinel-1 SAR analysis complete</span>'
            f'</div>\n\n'
            f"**Oil Probability (Stage 1):** `{oil_prob:.1f}%`\n\n"
            f"---\n\n"
            f"**Predicted Class:** `{pred_class}`\n\n"
            f"**Confidence:** `{confidence:.1f}%`\n\n"
            f"> {pred_desc}\n\n"
            f"---\n\n"
            f"**Class Probabilities (Stage 2)**\n\n"
            f"| Class | Probability | Status |\n"
            f"|---|---|---|\n"
            f"{rows}\n"
            f"{noaa_section}"
            f"*Device: {DEVICE.upper()}*"
        )
        return result

    except Exception as e:
        return f"**Error during inference:**\n\n```\n{e}\n```"


# ---------------------------------------------------------------------------
# Two-Stage Inference — batch (multiple images)
# ---------------------------------------------------------------------------
def predict_batch(images: list):
    """Run two-stage prediction on a list of uploaded images."""
    if not images:
        return "Please upload one or more Sentinel-1 SAR images."

    oil_model       = get_oil_model()
    thickness_model = get_thickness_model()
    results_parts   = []
    oil_count       = 0
    no_oil_count    = 0

    for idx, img_data in enumerate(images, start=1):
        try:
            # Gradio Gallery returns (PIL.Image, caption) tuples
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

            # ── Stage 1: Oil / No-Oil ──
            with torch.no_grad():
                oil_logits = oil_model(tensor)
            oil_probs = torch.softmax(oil_logits, dim=1).squeeze(0).cpu().numpy()
            oil_prob  = float(oil_probs[1]) * 100
            is_oil    = oil_prob >= (OIL_THRESHOLD * 100)

            if not is_oil:
                # No Oil detected — skip Stage 2
                no_oil_count += 1
                part = (
                    f'<div style="background:#E8F5E9; border-left:5px solid #2E7D32; '
                    f'padding:12px 16px; border-radius:8px; margin-bottom:10px;">'
                    f'<span style="font-size:1.15rem; font-weight:800; color:#2E7D32;">'
                    f'✅ Image {idx}: {filename}</span><br>'
                    f'<span style="font-size:0.82rem; color:#333;">NO OIL DETECTED</span>'
                    f'</div>\n\n'
                    f"**Oil Probability:** `{oil_prob:.1f}%` &nbsp;|&nbsp; "
                    f"**Stage 2:** *Not executed*\n\n"
                )
            else:
                # ── Stage 2: Thickness classification ──
                oil_count += 1
                with torch.no_grad():
                    thickness_logits = thickness_model(tensor)

                probs    = torch.softmax(thickness_logits, dim=1).squeeze(0).cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_class = CLASS_LABELS[pred_idx]
                confidence = float(probs[pred_idx]) * 100
                pred_desc  = CLASS_DESCS[pred_idx]

                rows = ""
                for i in range(len(CLASS_LABELS)):
                    cls  = CLASS_LABELS[i]
                    prob = f"{probs[i]*100:.2f}%"
                    if i == pred_idx:
                        rows += f"| **{cls}** | **{prob}** | ✅ Predicted |\n"
                    else:
                        rows += f"| {cls} | {prob} | |\n"

                # ── NOAA ADIOS oil match ──
                noaa = select_openoil_type_grounded(pred_class)
                noaa_line = ""
                if noaa.get("real_oil_name"):
                    noaa_line = f"**NOAA Match:** {noaa['real_oil_name']} (API {noaa.get('api_gravity', 'N/A')}) — `{noaa.get('opendrift_oiltype', '')}`\n\n"
                else:
                    noaa_line = f"**NOAA Match:** `{noaa.get('opendrift_oiltype', 'N/A')}` (generic)\n\n"

                part = (
                    f'<div style="background:#FFEBEE; border-left:5px solid #B71C1C; '
                    f'padding:12px 16px; border-radius:8px; margin-bottom:10px;">'
                    f'<span style="font-size:1.15rem; font-weight:800; color:#B71C1C;">'
                    f'🛢️ Image {idx}: {filename}</span><br>'
                    f'<span style="font-size:0.82rem; color:#333;">OIL DETECTED</span>'
                    f'</div>\n\n'
                    f"**Oil Probability:** `{oil_prob:.1f}%`\n\n"
                    f"**Predicted Class:** `{pred_class}` &nbsp;|&nbsp; "
                    f"**Confidence:** `{confidence:.1f}%`\n\n"
                    f"> {pred_desc}\n\n"
                    f"| Class | Probability | Status |\n"
                    f"|---|---|---|\n"
                    f"{rows}\n"
                    f"{noaa_line}"
                )

            results_parts.append(part)

        except Exception as e:
            results_parts.append(
                f"### Image {idx}\n\n"
                f"**Error:** `{e}`\n\n"
            )

    # Summary header with oil/no-oil counts
    header = (
        f'<div style="background:#E8F5E9; border-left:5px solid #2E7D32; '
        f'padding:12px 16px; border-radius:8px; margin-bottom:16px;">'
        f'<span style="font-size:1.3rem; font-weight:800; color:#1B5E20;">'
        f'Batch Analysis Complete — {len(images)} image(s) processed</span><br>'
        f'<span style="font-size:0.9rem; color:#333;">'
        f'🛢️ Oil: {oil_count} &nbsp;|&nbsp; ✅ No Oil: {no_oil_count}</span>'
        f'</div>\n\n'
    )

    separator = "\n---\n\n"
    return header + separator.join(results_parts) + f"\n*Device: {DEVICE.upper()}*"


# ---------------------------------------------------------------------------
# Gradio theme -- force light white + green
# ---------------------------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("Roboto Mono"),
).set(
    body_background_fill="#FFFFFF",
    body_background_fill_dark="#FFFFFF",
    body_text_color="#1B5E20",
    body_text_color_dark="#1B5E20",
    body_text_color_subdued="#333333",
    body_text_color_subdued_dark="#333333",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#C8E6C9",
    block_border_color_dark="#C8E6C9",
    block_label_text_color="#1B5E20",
    block_label_text_color_dark="#1B5E20",
    block_title_text_color="#1B5E20",
    block_title_text_color_dark="#1B5E20",
    button_primary_background_fill="linear-gradient(135deg, #2E7D32, #388E3C)",
    button_primary_background_fill_dark="linear-gradient(135deg, #2E7D32, #388E3C)",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_text_color="#2E7D32",
    button_secondary_text_color_dark="#2E7D32",
    button_secondary_border_color="#43A047",
    button_secondary_border_color_dark="#43A047",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#C8E6C9",
    input_border_color_dark="#C8E6C9",
    panel_background_fill="#FFFFFF",
    panel_background_fill_dark="#FFFFFF",
    panel_border_color="#C8E6C9",
    panel_border_color_dark="#C8E6C9",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#F9FBF9",
    background_fill_secondary_dark="#F9FBF9",
    border_color_primary="#C8E6C9",
    border_color_primary_dark="#C8E6C9",
    color_accent="#2E7D32",
    color_accent_soft="#E8F5E9",
    color_accent_soft_dark="#E8F5E9",
)

# ---------------------------------------------------------------------------
# Extra CSS (on top of theme)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ====== Header banner ====== */
#header-banner {
    background: linear-gradient(135deg, #1B5E20 0%, #388E3C 100%);
    border-radius: 14px;
    padding: 32px 36px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
#header-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        45deg, transparent, transparent 40px,
        rgba(255,255,255,0.04) 40px, rgba(255,255,255,0.04) 80px
    );
    pointer-events: none;
}
#header-banner h1 { color: #fff !important; font-size: 1.8rem !important;
    font-weight: 700 !important; margin: 0 0 4px !important; }
#header-banner p  { color: #C8E6C9 !important; font-size: 0.92rem !important;
    margin: 0 !important; }
#header-banner span { color: #C8E6C9 !important; }

/* ====== Force all body text dark green ====== */
.gradio-container, .gradio-container *,
.prose, .prose *, label, span, p, h1, h2, h3, h4 {
    color: #1B5E20 !important;
}
/* Re-override header */
#header-banner h1, #header-banner p, #header-banner span,
#header-banner *, #header-banner div {
    color: #fff !important;
}

/* ====== Button text white ====== */
#predict-btn, #predict-btn span,
#batch-btn, #batch-btn span { color: #fff !important; }

/* ====== Badges ====== */
.badge {
    display: inline-block;
    background: #E8F5E9;
    color: #1B5E20 !important;
    border: 1px solid #C8E6C9;
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* ====== Footer ====== */
#footer-text {
    text-align: center;
    color: #388E3C !important;
    font-size: 0.82rem !important;
    margin-top: 12px;
    font-style: italic;
}

/* ====== Output markdown styling ====== */
.output-md table { width: 100%; border-collapse: collapse; }
.output-md th {
    background: #E8F5E9 !important;
    color: #1B5E20 !important;
    font-weight: 700 !important;
    padding: 8px 12px; text-align: left;
    border-bottom: 2px solid #C8E6C9;
}
.output-md td {
    padding: 8px 12px; text-align: left;
    border-bottom: 1px solid #E8F5E9;
    color: #1B5E20 !important;
}
.output-md code {
    background: #E8F5E9 !important;
    color: #2E7D32 !important;
    border-radius: 4px; padding: 2px 6px;
    font-weight: 600;
}
.output-md blockquote {
    border-left: 3px solid #43A047;
    padding-left: 12px;
    color: #333 !important;
    font-style: italic;
}

/* Hide Gradio footer */
footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks() as demo:

        # Header
        gr.HTML("""
        <div id="header-banner">
            <h1>SAR Oil Spill Thickness Classifier</h1>
            <p>Two-Stage CNN + Swin Transformer Pipeline &nbsp;&middot;&nbsp;
               <span>Sentinel-1 SAR Image &rarr; Oil Detection &rarr; Thickness Class</span>
            </p>
        </div>
        """)

        # How it works
        with gr.Accordion("How it works", open=False):
            gr.Markdown("""
**Two-Stage Hierarchical Inference Pipeline**

This application uses **two separate models** in sequence:

**Stage 1 — Oil / No-Oil Detection**

| Detail | Value |
|---|---|
| Task | Binary classification: Oil vs No-Oil |
| Output | Oil probability (%) |
| Action | If No Oil → STOP. If Oil → proceed to Stage 2 |

**Stage 2 — Oil Thickness Classification** *(only if Stage 1 detects oil)*

| Class | Meaning |
|---|---|
| **Thin_Sheen** | Thin oil-sheen-like SAR signature -- light surface film |
| **Moderate** | Moderate thickness / emulsion SAR signature |
| **Thick_Emulsified** | Thick or emulsified oil -- heavy backscatter suppression |

**Input:** Sentinel-1 SAR image (resized to 224×224 internally)

**Model Architecture (shared backbone)**

| Component | Details |
|---|---|
| CNN Branch | ResNet-18 backbone -- 512-dim features |
| Swin Branch | Swin-Tiny (patch 4, window 7, 224×224) -- 768-dim features |
| Stage 1 Fusion | Concat 1280 → Linear → BN → ReLU → Dropout → Linear → ReLU → Dropout → Linear(2) |
| Stage 2 Fusion | Concat 1280 → Linear → BN → ReLU → Linear(3) |
            """)

        # Badges (shared)
        device_label = "GPU (CUDA)" if DEVICE == "cuda" else "CPU"

        # ===================== Tabs =====================
        with gr.Tabs():

            # -------- Tab 1: Single Image --------
            with gr.TabItem("🖼️ Single Image"):
                with gr.Row(equal_height=True):
                    # Left column: input
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Image")
                        image_input = gr.Image(
                            type="pil",
                            label="Upload SAR Image",
                            height=340,
                        )
                        with gr.Row():
                            predict_btn = gr.Button(
                                "Analyse SAR Image",
                                elem_id="predict-btn",
                                variant="primary",
                            )
                            clear_btn = gr.ClearButton(
                                [image_input],
                                value="Clear",
                                variant="secondary",
                            )

                    # Right column: output
                    with gr.Column(scale=1):
                        gr.Markdown("### Prediction Results")
                        output_md = gr.Markdown(
                            value="*Upload a Sentinel-1 SAR image and click **Analyse SAR Image** to detect oil and classify its thickness.*",
                            elem_classes=["output-md"],
                        )
                        gr.HTML(f"""
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            <span class="badge">Device: {device_label}</span>
                            <span class="badge">Pipeline: Two-Stage</span>
                            <span class="badge">Stage 1: Oil/No-Oil</span>
                            <span class="badge">Stage 2: Thin / Moderate / Thick</span>
                        </div>
                        """)

                # Examples (single-image tab)
                example_dir = os.path.dirname(os.path.abspath(__file__))
                example_images = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
                    example_images += glob.glob(os.path.join(example_dir, ext))
                if example_images:
                    gr.Examples(
                        examples=[[img] for img in example_images[:6]],
                        inputs=image_input,
                        label="Sample SAR Images",
                    )

                # Single-image events
                predict_btn.click(fn=predict, inputs=[image_input], outputs=[output_md],
                                  api_name="predict")
                image_input.upload(fn=predict, inputs=[image_input], outputs=[output_md])

            # -------- Tab 2: Batch / Multiple Images --------
            with gr.TabItem("📂 Batch Analysis"):
                gr.Markdown(
                    "### Upload Multiple SAR Images\n"
                    "Select **multiple images** at once and run batch analysis on all of them."
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        batch_gallery = gr.Gallery(
                            label="Upload SAR Images (drag & drop or click to select multiple)",
                            type="pil",
                            columns=3,
                            height=360,
                            object_fit="contain",
                        )
                        with gr.Row():
                            batch_btn = gr.Button(
                                "Analyse All Images",
                                elem_id="batch-btn",
                                variant="primary",
                            )
                            batch_clear_btn = gr.ClearButton(
                                [batch_gallery],
                                value="Clear All",
                                variant="secondary",
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Results")
                        batch_output_md = gr.Markdown(
                            value="*Upload multiple SAR images and click **Analyse All Images** to run batch predictions.*",
                            elem_classes=["output-md"],
                        )
                        gr.HTML(f"""
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            <span class="badge">Device: {device_label}</span>
                            <span class="badge">Mode: Batch</span>
                            <span class="badge">Pipeline: Two-Stage</span>
                        </div>
                        """)

                # Batch events
                batch_btn.click(fn=predict_batch, inputs=[batch_gallery],
                                outputs=[batch_output_md], api_name="predict_batch")

        # Footer
        gr.HTML("""
        <p id="footer-text">
            CNN-Swin SAR Oil Thickness Classifier &nbsp;&middot;&nbsp;
            Built with Gradio
        </p>
        """)

    return demo


# ---------------------------------------------------------------------------
# Build demo at module level (required for HF Spaces Gradio SDK)
# ---------------------------------------------------------------------------
print(f"PyTorch: {torch.__version__}")
print(f"Device : {DEVICE}")

print("Loading Stage 1 model (Oil/No-Oil detector)...")
get_oil_model()
print("[OK] Stage 1 model loaded successfully!")

print("Loading Stage 2 model (Thickness classifier)...")
get_thickness_model()
print("[OK] Stage 2 model loaded successfully!")

demo = build_app()

if __name__ == "__main__":
    print("Launching Gradio app...\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=False,
        css=CUSTOM_CSS,
        theme=custom_theme,
    )
