"""
app.py -- Professional Gradio UI for Hierarchical CNN-Swin SAR Oil Detection & Thickness Classifier

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
        return "<p style='color:#ef4444; font-weight:600; padding:12px;'>Please upload a Sentinel-1 SAR image first.</p>"
    try:
        tensor = preprocess(image)
        oil_detector, thickness_model = get_models()

        # Stage 1: Binary Oil Detection
        with torch.no_grad():
            oil_logits = oil_detector(tensor)
            oil_probs  = torch.softmax(oil_logits, dim=1).squeeze(0).cpu().numpy()

        p_no_oil = float(oil_probs[0])
        p_oil    = float(oil_probs[1])
        is_oil   = p_oil >= 0.5

        if is_oil:
            # Stage 2: Thickness Classification
            with torch.no_grad():
                thick_logits = thickness_model(tensor)
                thick_probs  = torch.softmax(thick_logits, dim=1).squeeze(0).cpu().numpy()

            pred_idx   = int(np.argmax(thick_probs))
            pred_class = THICKNESS_CLASSES[pred_idx]
            thick_conf = float(thick_probs[pred_idx]) * 100
            pred_desc  = CLASS_DESCS[pred_class]

            noaa_info = select_openoil_type_grounded(pred_class)
            noaa_name = noaa_info.get("real_oil_name", "N/A")
            noaa_api  = noaa_info.get("api_gravity", "N/A")
            noaa_visc = noaa_info.get("viscosity_cSt", "N/A")

            rows_html = ""
            for i, cls in enumerate(THICKNESS_CLASSES):
                prob_pct = thick_probs[i] * 100
                is_best = (i == pred_idx)
                row_bg = "background: rgba(220, 38, 38, 0.12);" if is_best else ""
                badge = "<span style='color:#ef4444; font-weight:700;'>[Predicted Class]</span>" if is_best else ""
                rows_html += f"""
                <tr style='{row_bg} border-bottom: 1px solid rgba(255,255,255,0.08);'>
                    <td style='padding: 10px 14px; font-weight: 600;'>{cls}</td>
                    <td style='padding: 10px 14px; font-weight: 700;'>{prob_pct:.2f}%</td>
                    <td style='padding: 10px 14px;'>{badge}</td>
                </tr>
                """

            html = f"""
            <div style="font-family: system-ui, -apple-system, sans-serif;">
                <!-- Stage 1 Banner -->
                <div style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 4px 14px rgba(153, 27, 27, 0.3);">
                    <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #fca5a5;">Stage 1: Binary Detection Result</div>
                    <div style="font-size: 1.7rem; font-weight: 900; margin: 4px 0 2px 0; color: #ffffff;">OIL SPILL DETECTED</div>
                    <div style="font-size: 0.95rem; color: #fecaca;">Detection Confidence: <b>{p_oil*100:.1f}%</b> &nbsp;|&nbsp; Clean Sea Probability: {p_no_oil*100:.1f}%</div>
                </div>

                <!-- Stage 2 Banner -->
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px 24px; margin-bottom: 16px;">
                    <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #38bdf8;">Stage 2: Oil Thickness Classification</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin: 4px 0;">Class: <span style="color:#f87171;">{pred_class.replace('_', ' ')}</span> ({thick_conf:.1f}% confidence)</div>
                    <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.5; margin-top: 6px;">{pred_desc}</div>
                </div>

                <!-- Thickness Probabilities Table -->
                <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-bottom: 10px;">Stage 2 Thickness Probabilities</div>
                    <table style="width: 100%; border-collapse: collapse; color: #cbd5e1; font-size: 0.9rem;">
                        <thead>
                            <tr style="border-bottom: 2px solid #334155; text-align: left;">
                                <th style="padding: 10px 14px;">Thickness Class</th>
                                <th style="padding: 10px 14px;">Probability</th>
                                <th style="padding: 10px 14px;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>

                <!-- NOAA ADIOS Match -->
                <div style="background: #1e1b4b; border: 1px solid #3730a3; border-radius: 12px; padding: 16px;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #c7d2fe; margin-bottom: 8px;">NOAA ADIOS Grounded Oil Properties</div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; font-size: 0.88rem; color: #e0e7ff;">
                        <div><b>Mapped Oil:</b> <br><span style="color:#a5b4fc;">{noaa_name}</span></div>
                        <div><b>API Gravity:</b> <br><span style="color:#a5b4fc;">{noaa_api}°</span></div>
                        <div><b>Viscosity:</b> <br><span style="color:#a5b4fc;">{noaa_visc} cSt</span></div>
                    </div>
                </div>
            </div>
            """
            return html
        else:
            # Clean Sea
            html = f"""
            <div style="font-family: system-ui, -apple-system, sans-serif;">
                <div style="background: linear-gradient(135deg, #14532d 0%, #166534 100%); color: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 14px rgba(22, 101, 52, 0.3);">
                    <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #86efac;">Stage 1: Binary Detection Result</div>
                    <div style="font-size: 1.7rem; font-weight: 900; margin: 4px 0 6px 0; color: #ffffff;">CLEAN SEA (NON-OIL)</div>
                    <div style="font-size: 0.95rem; color: #bbf7d0;">Clean Sea Confidence: <b>{p_no_oil*100:.1f}%</b> &nbsp;|&nbsp; Oil Prob: {p_oil*100:.1f}%</div>
                    <div style="margin-top: 12px; font-size: 0.9rem; color: #dcfce7; line-height: 1.5;">No marine oil slick was detected in this Sentinel-1 SAR scene. Stage 2 thickness evaluation skipped.</div>
                </div>
            </div>
            """
            return html

    except Exception as e:
        return f"<div style='color:#ef4444; padding:12px; background:#7f1d1d; border-radius:8px;'><b>Error during inference:</b> {e}</div>"

# ---------------------------------------------------------------------------
# Batch Inference (Multiple Images)
# ---------------------------------------------------------------------------
@spaces.GPU
def predict_batch(images: list):
    if not images:
        return "<p style='color:#ef4444; font-weight:600; padding:12px;'>Please upload one or more Sentinel-1 SAR images.</p>"

    oil_detector, thickness_model = get_models()

    total_count = len(images)
    oil_count = 0
    clean_count = 0
    thick_counts = {cls: 0 for cls in THICKNESS_CLASSES}

    rows_data = []

    for idx, img_data in enumerate(images, start=1):
        try:
            if isinstance(img_data, tuple):
                pil_img = img_data[0]
                filename = img_data[1] if img_data[1] else f"Image_{idx}"
            elif isinstance(img_data, str):
                pil_img = Image.open(img_data)
                filename = os.path.basename(img_data)
            elif isinstance(img_data, Image.Image):
                pil_img = img_data
                filename = f"Image_{idx}"
            else:
                pil_img = Image.open(img_data)
                filename = getattr(img_data, 'name', f"Image_{idx}")

            if isinstance(filename, str) and (os.sep in filename or '/' in filename):
                filename = os.path.basename(filename)

            tensor = preprocess(pil_img)

            # Stage 1: Binary Detection
            with torch.no_grad():
                oil_logits = oil_detector(tensor)
                oil_probs  = torch.softmax(oil_logits, dim=1).squeeze(0).cpu().numpy()

            p_no_oil = float(oil_probs[0])
            p_oil    = float(oil_probs[1])
            is_oil   = p_oil >= 0.5

            if is_oil:
                oil_count += 1
                # Stage 2: Thickness Classification
                with torch.no_grad():
                    thick_logits = thickness_model(tensor)
                    thick_probs  = torch.softmax(thick_logits, dim=1).squeeze(0).cpu().numpy()

                pred_idx   = int(np.argmax(thick_probs))
                pred_class = THICKNESS_CLASSES[pred_idx]
                thick_conf = float(thick_probs[pred_idx]) * 100
                thick_counts[pred_class] += 1

                rows_data.append({
                    "idx": idx,
                    "filename": filename,
                    "is_oil": True,
                    "det_conf": p_oil * 100,
                    "thick_class": pred_class,
                    "thick_conf": thick_conf,
                })
            else:
                clean_count += 1
                rows_data.append({
                    "idx": idx,
                    "filename": filename,
                    "is_oil": False,
                    "det_conf": p_no_oil * 100,
                    "thick_class": "-",
                    "thick_conf": None,
                })

        except Exception as e:
            rows_data.append({
                "idx": idx,
                "filename": f"Image_{idx}",
                "is_oil": False,
                "det_conf": 0.0,
                "thick_class": f"Error: {e}",
                "thick_conf": None,
            })

    # Summary Cards HTML
    oil_pct   = (oil_count / total_count * 100) if total_count > 0 else 0
    clean_pct = (clean_count / total_count * 100) if total_count > 0 else 0

    table_rows_html = ""
    for r in rows_data:
        if r["is_oil"]:
            status_badge = "<span style='background:#7f1d1d; color:#fecaca; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>OIL DETECTED</span>"
            if r["thick_class"] == "Thin_Sheen":
                thick_badge = "<span style='background:#7c2d12; color:#ffedd5; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>Thin Sheen</span>"
            elif r["thick_class"] == "Moderate":
                thick_badge = "<span style='background:#713f12; color:#fef08a; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>Moderate</span>"
            else:
                thick_badge = "<span style='background:#991b1b; color:#fecaca; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>Thick Emulsified</span>"
            thick_conf_str = f"<b>{r['thick_conf']:.1f}%</b>"
        else:
            status_badge = "<span style='background:#14532d; color:#bbf7d0; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.8rem;'>CLEAN SEA</span>"
            thick_badge = "<span style='color:#64748b; font-size:0.85rem;'>-</span>"
            thick_conf_str = "<span style='color:#64748b; font-size:0.85rem;'>-</span>"

        table_rows_html += f"""
        <tr style="border-bottom: 1px solid #1e293b;">
            <td style="padding: 10px 14px; font-weight: 600; color: #94a3b8;">#{r['idx']}</td>
            <td style="padding: 10px 14px; font-weight: 600; color: #f8fafc;">{r['filename']}</td>
            <td style="padding: 10px 14px;">{status_badge}</td>
            <td style="padding: 10px 14px; font-weight: 700; color: #e2e8f0;">{r['det_conf']:.1f}%</td>
            <td style="padding: 10px 14px;">{thick_badge}</td>
            <td style="padding: 10px 14px; color: #e2e8f0;">{thick_conf_str}</td>
        </tr>
        """

    html = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif;">
        <!-- Top Metrics Grid -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 18px;">
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px; text-align: center;">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Total Processed</div>
                <div style="color: #f8fafc; font-size: 2.2rem; font-weight: 900; margin-top: 4px;">{total_count}</div>
                <div style="color: #64748b; font-size: 0.8rem;">SAR Scenes</div>
            </div>

            <div style="background: #1e293b; border: 1px solid #dc2626; border-radius: 12px; padding: 18px; text-align: center;">
                <div style="color: #f87171; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Oil Spills</div>
                <div style="color: #ef4444; font-size: 2.2rem; font-weight: 900; margin-top: 4px;">{oil_count}</div>
                <div style="color: #fca5a5; font-size: 0.8rem; font-weight: 600;">{oil_pct:.1f}% of total</div>
            </div>

            <div style="background: #1e293b; border: 1px solid #16a34a; border-radius: 12px; padding: 18px; text-align: center;">
                <div style="color: #4ade80; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Clean Sea</div>
                <div style="color: #22c55e; font-size: 2.2rem; font-weight: 900; margin-top: 4px;">{clean_count}</div>
                <div style="color: #86efac; font-size: 0.8rem; font-weight: 600;">{clean_pct:.1f}% of total</div>
            </div>
        </div>

        <!-- Oil Thickness Distribution Banner -->
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="color: #e2e8f0; font-size: 0.95rem; font-weight: 700; margin-bottom: 10px;">Oil Thickness Breakdown (for {oil_count} detected spills)</div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: #7c2d12; color: #ffedd5; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #ea580c;">
                    Thin Sheen: <b>{thick_counts['Thin_Sheen']}</b>
                </span>
                <span style="background: #713f12; color: #fef08a; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #ca8a04;">
                    Moderate: <b>{thick_counts['Moderate']}</b>
                </span>
                <span style="background: #7f1d1d; color: #fecaca; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #dc2626;">
                    Thick Emulsified: <b>{thick_counts['Thick_Emulsified']}</b>
                </span>
            </div>
        </div>

        <!-- Detailed Results Table -->
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            <div style="background: #1e293b; padding: 14px 18px; font-weight: 700; color: #f8fafc; font-size: 0.95rem; border-bottom: 1px solid #334155;">
                Detailed Scene Analysis Results
            </div>
            <div style="max-height: 480px; overflow-y: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; text-align: left;">
                    <thead style="background: #090d16; color: #94a3b8; position: sticky; top: 0; z-index: 2;">
                        <tr style="border-bottom: 2px solid #1e293b;">
                            <th style="padding: 12px 14px;">#</th>
                            <th style="padding: 12px 14px;">Scene Filename</th>
                            <th style="padding: 12px 14px;">Stage 1 Status</th>
                            <th style="padding: 12px 14px;">Stage 1 Conf</th>
                            <th style="padding: 12px 14px;">Stage 2 Thickness</th>
                            <th style="padding: 12px 14px;">Thickness Conf</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """
    return html

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
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    color: #FFFFFF !important;
    padding: 24px 32px;
    border-radius: 14px;
    margin-bottom: 20px;
    border: 1px solid #334155;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
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
    color: #94A3B8 !important;
    margin: 0;
}
.badge {
    display: inline-block;
    background: #1E293B;
    color: #38BDF8;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #334155;
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
            <h1>Hierarchical SAR Oil Spill Detector & Thickness Classifier</h1>
            <p>Stage 1: Binary Oil Detection (cnn_swin_v2_best.pth) &nbsp;&middot;&nbsp;
               Stage 2: 3-Class Thickness Classification (cnn_swin_thickness_best.pth)
            </p>
        </div>
        """)

        # Architecture Explainer
        with gr.Accordion("Hierarchical Pipeline Architecture & Thickness Classes", open=False):
            gr.Markdown("""
### Hierarchical Pipeline Architecture

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

### Thickness Classes

| Class | Description | SAR Signature |
|---|---|---|
| **Thin_Sheen** | Light surface film, minimal impact | Faint backscatter dampening |
| **Moderate** | Intermediate emulsion, notable extent | Clear SAR signature |
| **Thick_Emulsified** | Heavy emulsion, immediate response needed | Strong backscatter suppression |

**Input**: Sentinel-1 SAR image normalized to `(3, 224, 224)`
            """)

        device_label = "GPU (CUDA)" if DEVICE == "cuda" else "CPU"

        with gr.Tabs():
            # Tab 1: Single Image
            with gr.TabItem("Single Image Analysis"):
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
                                "Run Detection & Thickness Analysis",
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
                        output_html = gr.HTML(
                            value="<p style='color:#94a3b8; font-style:italic;'>Upload a Sentinel-1 SAR image and click <b>Run Detection & Thickness Analysis</b>.</p>",
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

                predict_btn.click(fn=predict, inputs=[image_input], outputs=[output_html], api_name="predict")
                image_input.upload(fn=predict, inputs=[image_input], outputs=[output_html])

            # Tab 2: Batch Analysis
            with gr.TabItem("Batch SAR Processing"):
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
                            batch_btn = gr.Button("Analyze All Images", variant="primary")
                            batch_clear_btn = gr.ClearButton([batch_gallery], value="Clear All", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Summary & Results")
                        batch_output_html = gr.HTML(
                            value="<p style='color:#94a3b8; font-style:italic;'>Upload multiple SAR images and click <b>Analyze All Images</b>.</p>",
                        )
                        gr.HTML(f"""
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            <span class="badge">Device: {device_label}</span>
                            <span class="badge">Mode: Hierarchical Batch Pipeline</span>
                        </div>
                        """)

                batch_btn.click(fn=predict_batch, inputs=[batch_gallery], outputs=[batch_output_html], api_name="predict_batch")

        # Footer
        gr.HTML("""
        <p style="text-align:center; font-size:0.85rem; color:#64748b; margin-top:24px;">
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
print("[Init] Both models loaded successfully")

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
