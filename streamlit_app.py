"""
streamlit_app.py — Streamlit UI for Two-Stage SAR Oil Spill Detector & Thickness Classifier

Two-stage hierarchical inference pipeline:
  Stage 1: Oil / No-Oil detection    (cnn_swin_v2_best.pth  → 2 classes)
  Stage 2: Oil thickness classifier   (cnn_swin_thickness_best.pth → 3 classes)
           (Runs ONLY if Stage 1 detects oil)
"""

import os
import io
import time
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from model import load_model, load_oil_detector
from noaa_oils import select_openoil_type_grounded, get_adios_summary

# =========================================================================
# Configuration & Constants
# =========================================================================
CKPT_OIL_DETECTOR = os.getenv("CKPT_OIL_DETECTOR", "cnn_swin_v2_best.pth")
CKPT_THICKNESS = os.getenv("CKPT_THICKNESS", "cnn_swin_thickness_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THICKNESS_CLASSES = ["Thin_Sheen", "Moderate", "Thick_Emulsified"]
THICKNESS_COLORS = {
    "Thin_Sheen": "#00d2ff",
    "Moderate": "#f5a623",
    "Thick_Emulsified": "#ff4757",
}
THICKNESS_DESCRIPTIONS = {
    "Thin_Sheen": "Very low thickness / surface sheen. High rate of natural dispersion. Minimal skimming requirement.",
    "Moderate": "Intermediate thickness / emulsion. Mechanical containment (booms/skimmers) recommended.",
    "Thick_Emulsified": "Heavy emulsified slick. Immediate response required — high threat to marine ecosystems.",
}

# Standard ImageNet normalization for 224x224 input
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# =========================================================================
# Page Setup & Styling
# =========================================================================
st.set_page_config(
    page_title="SAR Oil Spill Detection & Thickness Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #8892b0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .card-clean {
        background: linear-gradient(135deg, rgba(38, 166, 154, 0.15) 0%, rgba(0, 137, 123, 0.05) 100%);
        border: 1px solid rgba(38, 166, 154, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .card-oil {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.15) 0%, rgba(235, 47, 6, 0.05) 100%);
        border: 1px solid rgba(255, 71, 87, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .status-badge-oil {
        display: inline-block;
        background: rgba(255, 71, 87, 0.25);
        color: #ff6b81;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #ff4757;
        font-size: 1.1rem;
    }
    .status-badge-clean {
        display: inline-block;
        background: rgba(46, 213, 115, 0.2);
        color: #2ed573;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #2ed573;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Model Loader (Cached with Auto-Download Fallback)
# =========================================================================
import urllib.request

def download_file_with_progress(url: str, output_path: str, desc: str):
    """Downloads a checkpoint with a visible Streamlit status indicator."""
    progress_bar = st.progress(0, text=f"Downloading {desc} (one-time setup)...")
    try:
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(int(block_num * block_size * 100 / total_size), 100)
                progress_bar.progress(percent, text=f"Downloading {desc}: {percent}% ({int(block_num * block_size / (1024*1024))}MB / {int(total_size / (1024*1024))}MB)")
        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        progress_bar.empty()
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error downloading {desc}: {e}")
        raise e

@st.cache_resource(show_spinner="Loading Hybrid Deep Learning Models...")
def get_models():
    """Load and cache Stage 1 and Stage 2 models with fallback download."""
    # URLs for cloud deployment fallback if files aren't in git repo
    DOWNLOAD_URLS = {
        "cnn_swin_v2_best.pth": os.getenv("URL_OIL_DETECTOR", ""),
        "cnn_swin_thickness_best.pth": os.getenv("URL_THICKNESS", ""),
    }

    # Check Stage 1 detector
    if not os.path.exists(CKPT_OIL_DETECTOR):
        url = DOWNLOAD_URLS.get(os.path.basename(CKPT_OIL_DETECTOR))
        if url:
            download_file_with_progress(url, CKPT_OIL_DETECTOR, "Stage 1 Oil Detector")
        else:
            raise FileNotFoundError(f"Oil detector checkpoint not found at '{CKPT_OIL_DETECTOR}'. Place it in the directory or set URL_OIL_DETECTOR environment variable.")

    # Check Stage 2 thickness classifier (main model)
    if not os.path.exists(CKPT_THICKNESS):
        url = DOWNLOAD_URLS.get(os.path.basename(CKPT_THICKNESS))
        if url:
            download_file_with_progress(url, CKPT_THICKNESS, "Stage 2 Thickness Model")
        else:
            raise FileNotFoundError(f"Thickness checkpoint not found at '{CKPT_THICKNESS}'. Place it in the directory or set URL_THICKNESS environment variable.")

    oil_detector = load_oil_detector(CKPT_OIL_DETECTOR, device=DEVICE)
    thickness_model = load_model(CKPT_THICKNESS, device=DEVICE)
    return oil_detector, thickness_model


# =========================================================================
# Inference Pipeline
# =========================================================================
def run_two_stage_inference(pil_img: Image.Image, oil_model, thickness_model, oil_threshold=0.5):
    """
    Run hierarchical inference on a single PIL image.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    # --- Stage 1: Oil Detector ---
    with torch.no_grad():
        oil_logits = oil_model(tensor)
        oil_probs = F.softmax(oil_logits, dim=1).cpu().squeeze(0).numpy()

    p_no_oil = float(oil_probs[0])
    p_oil = float(oil_probs[1])
    is_oil = p_oil >= oil_threshold

    result = {
        "is_oil": is_oil,
        "oil_probability": p_oil,
        "no_oil_probability": p_no_oil,
        "thickness_class": None,
        "thickness_confidence": None,
        "thickness_distribution": {},
        "stage2_executed": False,
    }

    # --- Stage 2: Thickness Classifier (Only if oil detected) ---
    if is_oil:
        with torch.no_grad():
            thick_logits = thickness_model(tensor)
            thick_probs = F.softmax(thick_logits, dim=1).cpu().squeeze(0).numpy()

        best_idx = int(np.argmax(thick_probs))
        result["thickness_class"] = THICKNESS_CLASSES[best_idx]
        result["thickness_confidence"] = float(thick_probs[best_idx])
        result["thickness_distribution"] = {
            cls: float(thick_probs[i]) for i, cls in enumerate(THICKNESS_CLASSES)
        }
        result["stage2_executed"] = True

    return result


# =========================================================================
# Sidebar
# =========================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite.png", width=64)
    st.title("System Control")
    st.markdown(f"**Compute Device:** `{DEVICE.upper()}`")
    
    st.markdown("---")
    st.subheader("⚙️ Detection Threshold")
    oil_thresh = st.slider(
        "Stage 1 Oil Sensitivity Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability cutoff for Stage 1 to classify the SAR scene as containing oil."
    )

    st.markdown("---")
    st.subheader("📦 Model Checkpoints")
    st.text(f"Stage 1: {os.path.basename(CKPT_OIL_DETECTOR)}")
    st.text(f"Stage 2: {os.path.basename(CKPT_THICKNESS)}")

    st.markdown("---")
    st.markdown(
        "**Pipeline Overview:**\n"
        "1. **Stage 1 (ResNet-18 + Swin-Tiny):** Binary Oil / Clean detection\n"
        "2. **Stage 2 (ResNet-18 + Swin-Tiny):** 3-Class Oil thickness estimation (Runs only if Oil is detected)"
    )

    st.markdown("---")
    st.subheader("🛢️ NOAA ADIOS Database")
    try:
        adios_summary = get_adios_summary()
        total = sum(adios_summary.values())
        if total > 0:
            st.success(f"✅ {total} real oils loaded")
            for cls, count in adios_summary.items():
                st.text(f"{cls.replace('_', ' ')}: {count} oils")
        else:
            st.warning("⚠️ ADIOS DB not yet downloaded\n(will clone on first oil detection)")
    except Exception:
        st.info("ADIOS DB: Loading on demand")


# =========================================================================
# Main App Interface
# =========================================================================
st.markdown('<div class="main-header">🛰️ Sentinel-1 SAR Oil Spill Inference</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Two-Stage Hierarchical AI System: Oil Detection & Thickness Characterization</div>', unsafe_allow_html=True)

try:
    oil_model, thickness_model = get_models()
except Exception as e:
    st.error(f"❌ Error loading model checkpoints: {e}")
    st.stop()

tab_single, tab_batch, tab_arch, tab_noaa = st.tabs(["🔍 Single Scene Analysis", "📁 Batch Processing", "🧠 Model Architecture", "🛢️ NOAA ADIOS Oils"])

# -------------------------------------------------------------------------
# TAB 1: Single Scene Analysis
# -------------------------------------------------------------------------
with tab_single:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Upload SAR Image")
        uploaded_file = st.file_uploader(
            "Select Sentinel-1 SAR file (.png, .jpg, .tif, .bmp)",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key="single_uploader"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded SAR Scene: {uploaded_file.name}", use_container_width=True)
        else:
            st.info("👆 Please upload a SAR image to begin analysis.")

    with col_output:
        st.subheader("2. Inference Diagnosis")

        if uploaded_file is not None:
            with st.spinner("Executing Two-Stage Deep Learning Pipeline..."):
                t0 = time.time()
                res = run_two_stage_inference(image, oil_model, thickness_model, oil_threshold=oil_thresh)
                elapsed_ms = (time.time() - t0) * 1000

            st.caption(f"⏱️ Inference completed in {elapsed_ms:.1f} ms")

            # Stage 1 Display
            st.markdown("#### Stage 1: Detection Output")
            if res["is_oil"]:
                st.markdown(f'''
                <div class="card-oil">
                    <span class="status-badge-oil">⚠️ OIL SPILL DETECTED</span>
                    <p style="margin-top: 10px; font-size: 1.1rem;">
                        Oil Probability: <b>{res["oil_probability"]*100:.2f}%</b> (No-Oil: {res["no_oil_probability"]*100:.2f}%)
                    </p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="card-clean">
                    <span class="status-badge-clean">✅ NO OIL DETECTED (CLEAN SEA / LOOKALIKE)</span>
                    <p style="margin-top: 10px; font-size: 1.1rem;">
                        Confidence (Clean): <b>{res["no_oil_probability"]*100:.2f}%</b> | Oil Prob: {res["oil_probability"]*100:.2f}%
                    </p>
                    <p style="color: #8892b0; margin-bottom: 0;">Stage 2 (Thickness Model) bypassed automatically.</p>
                </div>
                ''', unsafe_allow_html=True)

            # Stage 2 Display
            st.markdown("#### Stage 2: Characterization & Thickness")
            if res["stage2_executed"]:
                t_cls = res["thickness_class"]
                t_conf = res["thickness_confidence"] * 100
                t_color = THICKNESS_COLORS[t_cls]

                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 10px; border-left: 5px solid {t_color}; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: {t_color};">{t_cls.replace('_', ' ')}</h3>
                    <p style="font-size: 1.2rem; margin: 5px 0;">Confidence: <b>{t_conf:.2f}%</b></p>
                    <p style="color: #cbd5e1; font-size: 0.95rem; margin: 0;">{THICKNESS_DESCRIPTIONS[t_cls]}</p>
                </div>
                """, unsafe_allow_html=True)

                st.write("**Thickness Class Probability Breakdown:**")
                for cls, prob in res["thickness_distribution"].items():
                    col_name, col_bar = st.columns([1, 3])
                    with col_name:
                        st.write(f"**{cls.replace('_', ' ')}**")
                    with col_bar:
                        st.progress(prob, text=f"{prob*100:.2f}%")

                # ---- NOAA ADIOS Grounded Oil Recommendation ----
                st.markdown("#### 🛢️ NOAA ADIOS Grounded Oil Recommendation")
                with st.spinner("Looking up real NOAA-documented oil..."):
                    grounded = select_openoil_type_grounded(t_cls)

                if "real_oil_name" in grounded:
                    st.markdown(f"""
                    <div style="background: rgba(0,210,255,0.08); border: 1px solid rgba(0,210,255,0.3);
                                border-radius: 10px; padding: 16px; margin-top: 8px;">
                        <p style="margin:0; font-size:0.85rem; color:#8892b0;">NOAA ADIOS Real Oil Match</p>
                        <p style="margin:4px 0; font-size:1.15rem; font-weight:700; color:#00d2ff;">
                            🛢️ {grounded['real_oil_name']}
                        </p>
                        <table style="width:100%; font-size:0.9rem; color:#cbd5e1;">
                            <tr><td><b>API Gravity</b></td><td>{grounded.get('api_gravity', 'N/A')}</td></tr>
                            <tr><td><b>Source Location</b></td><td>{grounded.get('source_location', 'N/A')}</td></tr>
                            <tr><td><b>NOAA Labels</b></td><td>{', '.join(grounded.get('noaa_labels', []))}</td></tr>
                            <tr><td><b>OpenDrift Oil Type</b></td><td><code>{grounded['opendrift_oiltype']}</code></td></tr>
                        </table>
                        <p style="margin:8px 0 0; font-size:0.8rem; color:#8892b0;">
                            Fallback generic: <code>{grounded.get('fallback_generic_type', 'N/A')}</code>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"ℹ️ {grounded.get('notes', 'No ADIOS match found.')} Using: `{grounded['opendrift_oiltype']}`")
            else:
                st.info("ℹ️ Stage 2 was not triggered because no oil signature was detected in Stage 1.")


# -------------------------------------------------------------------------
# TAB 2: Batch Processing
# -------------------------------------------------------------------------
with tab_batch:
    st.subheader("📁 Batch Multi-Image Analysis")
    st.markdown("Upload multiple Sentinel-1 SAR tiles simultaneously for automated batch classification.")

    batch_files = st.file_uploader(
        "Select SAR images for batch processing",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if batch_files:
        st.write(f"**{len(batch_files)} image(s) uploaded.** Click below to run batch inference:")
        if st.button("🚀 Process Batch Images", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            records = []
            for i, file in enumerate(batch_files):
                status_text.text(f"Processing ({i+1}/{len(batch_files)}): {file.name}")
                try:
                    img = Image.open(file)
                    res = run_two_stage_inference(img, oil_model, thickness_model, oil_threshold=oil_thresh)
                    records.append({
                        "Filename": file.name,
                        "Oil Detected": "YES" if res["is_oil"] else "NO",
                        "Oil Probability (%)": round(res["oil_probability"] * 100, 2),
                        "Clean Probability (%)": round(res["no_oil_probability"] * 100, 2),
                        "Thickness Classification": res["thickness_class"].replace('_', ' ') if res["thickness_class"] else "N/A (No Oil)",
                        "Thickness Confidence (%)": round(res["thickness_confidence"] * 100, 2) if res["thickness_confidence"] else 0.0,
                    })
                except Exception as ex:
                    records.append({
                        "Filename": file.name,
                        "Oil Detected": "ERROR",
                        "Oil Probability (%)": 0.0,
                        "Clean Probability (%)": 0.0,
                        "Thickness Classification": str(ex),
                        "Thickness Confidence (%)": 0.0,
                    })
                progress_bar.progress((i + 1) / len(batch_files))

            status_text.success("✅ Batch analysis finished successfully!")
            
            df = pd.DataFrame(records)

            # Summary Metrics
            oil_count = sum(df["Oil Detected"] == "YES")
            clean_count = sum(df["Oil Detected"] == "NO")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Scenes", len(df))
            m2.metric("Oil Slicks Detected", oil_count, delta=f"{oil_count/len(df)*100:.1f}%")
            m3.metric("Clean Scenes", clean_count)

            st.markdown("#### Detailed Results Table")
            st.dataframe(df, use_container_width=True)

            # Download CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="sar_oil_spill_batch_results.csv",
                mime="text/csv",
            )


# -------------------------------------------------------------------------
# TAB 3: Model Architecture
# -------------------------------------------------------------------------
with tab_arch:
    st.subheader("🧠 Two-Stage CNN-Swin Deep Learning Architecture")
    st.markdown("""
    The system utilizes a **hierarchical dual-model architecture** designed to eliminate false positives on clean sea/lookalike scenes:

    ### Stage 1: Oil Detector (`cnn_swin_v2_best.pth`)
    - **Backbone 1 (CNN):** ResNet-18 extracting localized spatial feature maps (512-dim).
    - **Backbone 2 (Transformer):** Swin-Tiny Transformer capturing multi-scale global context & self-attention (768-dim).
    - **Fusion:** Concatenation (1280-dim) → Linear(1280, 512) → BatchNorm → ReLU → Dropout → Linear(512, 128) → ReLU → Dropout → Linear(128, 2).
    - **Task:** Binary classification (`No_Oil` vs `Oil`).

    ### Stage 2: Thickness Classifier (`cnn_swin_thickness_best.pth`)
    - **Trigger Condition:** Executed **only** when Stage 1 predicts oil presence above the threshold.
    - **Architecture:** ResNet-18 + Swin-Tiny hybrid with 3-class classification head.
    - **Classes:**
        - **Thin Sheen:** Low-thickness surface sheen.
        - **Moderate:** Intermediate thickness/emulsion.
        - **Thick Emulsified:** Heavy emulsion / concentrated oil slick.
    """)


# -------------------------------------------------------------------------
# TAB 4: NOAA ADIOS Oil Database Browser
# -------------------------------------------------------------------------
with tab_noaa:
    st.subheader("🛢️ NOAA ADIOS Oil Database — Thickness Class Browser")
    st.markdown("""
    This tab browses the **live NOAA ADIOS database** (1,400+ lab-tested oils), bucketed into the
    three SAR thickness classes using NOAA's own official labels — the same source used by
    **OpenDrift / OpenOil** for oil spill trajectory modelling.
    """)

    adios_summary = get_adios_summary()
    total_oils = sum(adios_summary.values())

    if total_oils == 0:
        st.warning("""
        ⚠️ **NOAA ADIOS database not yet downloaded.**

        The database will be cloned automatically (~300 MB, one-time) when you run an
        oil-detected image in the Single Scene tab, or you can trigger it manually below.
        """)
        if st.button("⬇️ Download NOAA ADIOS Database Now"):
            from noaa_oils import _ensure_adios_cloned, _load_adios, _adios_cache
            with st.spinner("Cloning NOAA ADIOS repository (~300 MB)..."):
                success = _ensure_adios_cloned()
            if success:
                st.success("✅ NOAA ADIOS database downloaded! Reload the page.")
                st.rerun()
            else:
                st.error("❌ Clone failed. Please check your internet connection and that git is installed.")
    else:
        st.success(f"✅ **{total_oils} real NOAA-documented oils** loaded from ADIOS database.")

        selected_class = st.selectbox(
            "Select Thickness Class to Browse:",
            ["Thin_Sheen", "Moderate", "Thick_Emulsified"],
            format_func=lambda x: x.replace("_", " "),
        )
        region_filter = st.text_input(
            "Filter by Region (optional)",
            placeholder="e.g. Gulf, Norway, Saudi Arabia, Alaska",
        )

        from noaa_oils import _load_adios
        db = _load_adios()
        oils = db.get(selected_class, [])
        if region_filter:
            oils = [o for o in oils if region_filter.lower() in o["location"].lower()]

        st.markdown(f"**{len(oils)} oil(s) found** for `{selected_class.replace('_', ' ')}`" +
                    (f" in regions matching *{region_filter}*" if region_filter else ""))

        import pandas as pd
        if oils:
            df_oils = pd.DataFrame([{
                "Oil Name":        o["name"],
                "API Gravity":     o.get("api", "N/A"),
                "Location":        o.get("location", "Unknown"),
                "NOAA Labels":     ", ".join(o.get("labels", [])),
            } for o in oils])
            st.dataframe(df_oils, use_container_width=True)

            csv_noaa = df_oils.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Oil List as CSV",
                data=csv_noaa,
                file_name=f"noaa_adios_{selected_class}.csv",
                mime="text/csv",
            )
        else:
            st.info("No oils found for the selected class/region filter.")
