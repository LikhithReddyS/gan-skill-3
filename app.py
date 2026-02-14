"""
DCGAN Cancer Image Generator — Streamlit App
A premium dashboard for visualizing training, generating images, and exploring the DCGAN model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob
from pathlib import Path
from PIL import Image

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from generator import Generator
from discriminator import Discriminator

# ─────────────────────────────────────────────────────
# Page Config & Custom CSS
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCGAN Cancer Image Generator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Gradient header bar ── */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
}
.main-header h1 {
    color: white;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin: 0.4rem 0 0 0;
    font-weight: 300;
}

/* ── Stat cards ── */
.stat-card {
    background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.18);
}
.stat-card .value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-card .label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}

/* ── Section dividers ── */
.section-divider {
    height: 3px;
    background: linear-gradient(90deg, #667eea, #f093fb, transparent);
    border: none;
    border-radius: 2px;
    margin: 1.5rem 0;
}

/* ── Image gallery cards ── */
.img-card {
    background: #1e1e2e;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    transition: transform 0.2s ease;
}
.img-card:hover {
    transform: scale(1.02);
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
}
[data-testid="stSidebar"] .stRadio label {
    color: #ccc;
    font-weight: 400;
}

/* ── Architecture code blocks ── */
.arch-block {
    background: #1a1a2e;
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px;
    padding: 1.2rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.85rem;
    color: #e0e0e0;
    overflow-x: auto;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.2);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.45) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 DCGAN Studio")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Training Dashboard", "🎨 Image Generator", "📸 Sample Browser", "🧠 Model Architecture"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#666;font-size:0.75rem;text-align:center;'>"
        "DCGAN · Cancer Histopathology<br>Built with Streamlit</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────
@st.cache_data
def load_training_logs():
    """Discover and load all training CSV logs."""
    log_dir = PROJECT_ROOT / "logs"
    logs = {}
    if log_dir.exists():
        for csv_path in sorted(log_dir.rglob("training_log.csv")):
            run_name = csv_path.parent.name
            try:
                df = pd.read_csv(csv_path)
                logs[run_name] = df
            except Exception:
                pass
    return logs


@st.cache_resource
def load_generator_from_checkpoint(checkpoint_path: str, latent_dim: int = 100):
    """Load a trained Generator from a .pt checkpoint."""
    device = torch.device("cpu")
    gen = Generator(latent_dim=latent_dim, feature_maps=64, channels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gen.load_state_dict(checkpoint["model_state_dict"])
    gen.eval()
    return gen


def generate_images(generator, num_images=8, seed=None):
    """Generate images using the Generator."""
    if seed is not None:
        torch.manual_seed(seed)
    device = next(generator.parameters()).device
    z = torch.randn(num_images, generator.latent_dim, device=device)
    with torch.no_grad():
        imgs = generator(z)
    # Convert from [-1,1] to [0,255] uint8
    imgs = ((imgs + 1) / 2).clamp(0, 1)
    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return imgs


def get_sample_images():
    """Get sorted list of sample image paths."""
    samples_dir = PROJECT_ROOT / "samples"
    if not samples_dir.exists():
        return []
    return sorted(samples_dir.glob("samples_epoch_*.png"), key=lambda p: p.name)


def get_checkpoints():
    """Get sorted list of Generator checkpoint paths."""
    cp_dir = PROJECT_ROOT / "checkpoints"
    if not cp_dir.exists():
        return []
    return sorted(cp_dir.glob("G_epoch_*.pt"), key=lambda p: p.name)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────
# Page: Training Dashboard
# ─────────────────────────────────────────────────────
def page_dashboard():
    st.markdown(
        "<div class='main-header'>"
        "<h1>📊 Training Dashboard</h1>"
        "<p>Visualize DCGAN training metrics across all runs</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    logs = load_training_logs()

    if not logs:
        st.warning("No training logs found in `logs/` directory. Train your model first!")
        return

    # Run selector
    run_names = list(logs.keys())
    selected_run = st.selectbox("Select Training Run", run_names, index=len(run_names) - 1)
    df = logs[selected_run]

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Summary stat cards ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{int(df['epoch'].max())}</div>"
            f"<div class='label'>Epochs</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        final_d = df["d_loss"].iloc[-1]
        st.markdown(
            f"<div class='stat-card'><div class='value'>{final_d:.3f}</div>"
            f"<div class='label'>Final D Loss</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        final_g = df["g_loss"].iloc[-1]
        st.markdown(
            f"<div class='stat-card'><div class='value'>{final_g:.3f}</div>"
            f"<div class='label'>Final G Loss</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        final_acc = df["d_real_acc"].iloc[-1]
        st.markdown(
            f"<div class='stat-card'><div class='value'>{final_acc:.1%}</div>"
            f"<div class='label'>D Real Acc</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Loss curves ──
    st.subheader("Loss Curves")
    loss_df = df[["epoch", "d_loss", "g_loss"]].set_index("epoch")
    loss_df.columns = ["Discriminator Loss", "Generator Loss"]
    st.line_chart(loss_df, use_container_width=True)

    # ── Accuracy curves ──
    st.subheader("Discriminator Accuracy")
    acc_df = df[["epoch", "d_real_acc", "d_fake_acc"]].set_index("epoch")
    acc_df.columns = ["Real Accuracy", "Fake Accuracy"]
    st.line_chart(acc_df, use_container_width=True)

    # ── Raw data expander ──
    with st.expander("📋 View Raw Metrics Table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────
# Page: Image Generator
# ─────────────────────────────────────────────────────
def page_generator():
    st.markdown(
        "<div class='main-header'>"
        "<h1>🎨 Image Generator</h1>"
        "<p>Generate synthetic cancer histopathology images from trained models</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    checkpoints = get_checkpoints()

    if not checkpoints:
        st.warning(
            "No Generator checkpoints found in `checkpoints/`. "
            "Train your model first to generate `.pt` files."
        )
        st.info("Run: `python src/train_dcgan.py --create_sample_data --epochs 10`")
        return

    # Controls
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        cp_names = [p.name for p in checkpoints]
        selected_cp = st.selectbox("Checkpoint", cp_names, index=len(cp_names) - 1)
    with col_b:
        num_images = st.slider("Number of Images", 1, 64, 16)
    with col_c:
        seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    if st.button("🚀 Generate Images", use_container_width=True):
        cp_path = PROJECT_ROOT / "checkpoints" / selected_cp
        with st.spinner("Loading model & generating..."):
            gen = load_generator_from_checkpoint(str(cp_path))
            imgs = generate_images(gen, num_images=num_images, seed=seed)

        st.success(f"Generated {num_images} images from **{selected_cp}**")

        # Display in grid
        cols_per_row = min(8, num_images)
        for row_start in range(0, num_images, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < num_images:
                    with col:
                        st.image(
                            imgs[idx],
                            caption=f"#{idx+1}",
                            use_container_width=True,
                        )


# ─────────────────────────────────────────────────────
# Page: Sample Browser
# ─────────────────────────────────────────────────────
def page_samples():
    st.markdown(
        "<div class='main-header'>"
        "<h1>📸 Sample Browser</h1>"
        "<p>Browse generated sample grids across training epochs</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    sample_paths = get_sample_images()

    if not sample_paths:
        st.warning("No sample images found in `samples/` directory.")
        return

    # Extract epoch numbers for the slider
    epoch_nums = []
    for p in sample_paths:
        try:
            num = int(p.stem.split("_")[-1])
            epoch_nums.append(num)
        except ValueError:
            pass

    if not epoch_nums:
        st.warning("Could not parse epoch numbers from sample filenames.")
        return

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Stats ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{len(sample_paths)}</div>"
            f"<div class='label'>Total Samples</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{min(epoch_nums)}</div>"
            f"<div class='label'>First Epoch</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{max(epoch_nums)}</div>"
            f"<div class='label'>Last Epoch</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── View mode ──
    view_mode = st.radio("View Mode", ["Single Epoch", "Side-by-Side Comparison", "Full Gallery"], horizontal=True)

    if view_mode == "Single Epoch":
        epoch = st.select_slider("Select Epoch", options=epoch_nums, value=epoch_nums[-1])
        idx = epoch_nums.index(epoch)
        img = Image.open(sample_paths[idx])
        st.image(img, caption=f"Epoch {epoch}", use_container_width=True)

    elif view_mode == "Side-by-Side Comparison":
        col_l, col_r = st.columns(2)
        with col_l:
            epoch_a = st.select_slider("Early Epoch", options=epoch_nums, value=epoch_nums[0], key="ep_a")
            idx_a = epoch_nums.index(epoch_a)
            st.image(Image.open(sample_paths[idx_a]), caption=f"Epoch {epoch_a}", use_container_width=True)
        with col_r:
            epoch_b = st.select_slider("Later Epoch", options=epoch_nums, value=epoch_nums[-1], key="ep_b")
            idx_b = epoch_nums.index(epoch_b)
            st.image(Image.open(sample_paths[idx_b]), caption=f"Epoch {epoch_b}", use_container_width=True)

    else:  # Full Gallery
        cols_per_row = st.slider("Columns", 1, 6, 3)
        for row_start in range(0, len(sample_paths), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(sample_paths):
                    with col:
                        img = Image.open(sample_paths[idx])
                        st.image(img, caption=f"Epoch {epoch_nums[idx]}", use_container_width=True)


# ─────────────────────────────────────────────────────
# Page: Model Architecture
# ─────────────────────────────────────────────────────
def page_architecture():
    st.markdown(
        "<div class='main-header'>"
        "<h1>🧠 Model Architecture</h1>"
        "<p>Generator & Discriminator architecture details</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    gen = Generator(latent_dim=100, feature_maps=64, channels=3)
    disc = Discriminator(channels=3, feature_maps=64, dropout=0.25)

    g_total, g_train = count_parameters(gen)
    d_total, d_train = count_parameters(disc)

    # ── Parameter stats ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{g_total:,}</div>"
            f"<div class='label'>Generator Params</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{d_total:,}</div>"
            f"<div class='label'>Discriminator Params</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{g_total + d_total:,}</div>"
            f"<div class='label'>Total Params</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='stat-card'><div class='value'>64×64</div>"
            f"<div class='label'>Image Size</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Architecture details ──
    tab_g, tab_d, tab_cfg = st.tabs(["🔵 Generator", "🔴 Discriminator", "⚙️ Training Config"])

    with tab_g:
        st.markdown("#### Generator Architecture")
        st.markdown(
            "Transforms a **100-dim latent vector** → **64×64 RGB image** via "
            "transposed convolutions with BatchNorm and ReLU."
        )
        st.code(str(gen), language="text")

        st.markdown("#### Data Flow")
        st.markdown(
            "```\n"
            "z (batch, 100)\n"
            "  → Linear + BN + ReLU  → (batch, 512×4×4)\n"
            "  → Reshape              → (batch, 512, 4, 4)\n"
            "  → ConvT(512→256) + BN + ReLU → (batch, 256, 8, 8)\n"
            "  → ConvT(256→128) + BN + ReLU → (batch, 128, 16, 16)\n"
            "  → ConvT(128→64)  + BN + ReLU → (batch, 64, 32, 32)\n"
            "  → ConvT(64→3)    + Tanh      → (batch, 3, 64, 64)\n"
            "```"
        )

    with tab_d:
        st.markdown("#### Discriminator Architecture")
        st.markdown(
            "Classifies **64×64 RGB images** as real or fake via "
            "strided convolutions with BatchNorm, LeakyReLU, and Dropout."
        )
        st.code(str(disc), language="text")

        st.markdown("#### Data Flow")
        st.markdown(
            "```\n"
            "x (batch, 3, 64, 64)\n"
            "  → Conv(3→64)   + LeakyReLU         → (batch, 64, 32, 32)\n"
            "  → Conv(64→128) + BN + LReLU + Drop  → (batch, 128, 16, 16)\n"
            "  → Conv(128→256)+ BN + LReLU + Drop  → (batch, 256, 8, 8)\n"
            "  → Conv(256→512)+ BN + LReLU + Drop  → (batch, 512, 4, 4)\n"
            "  → Conv(512→1)  + Sigmoid            → (batch, 1, 1, 1)\n"
            "```"
        )

    with tab_cfg:
        st.markdown("#### Training Configuration")

        # Load YAML configs if available
        train_cfg = PROJECT_ROOT / "configs" / "train_config.yaml"
        data_cfg = PROJECT_ROOT / "configs" / "data_config.yaml"

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("##### Training Config")
            if train_cfg.exists():
                st.code(train_cfg.read_text(), language="yaml")
            else:
                st.info("No train_config.yaml found.")
        with col_r:
            st.markdown("##### Data Config")
            if data_cfg.exists():
                st.code(data_cfg.read_text(), language="yaml")
            else:
                st.info("No data_config.yaml found.")


# ─────────────────────────────────────────────────────
# Route to page
# ─────────────────────────────────────────────────────
if page == "📊 Training Dashboard":
    page_dashboard()
elif page == "🎨 Image Generator":
    page_generator()
elif page == "📸 Sample Browser":
    page_samples()
elif page == "🧠 Model Architecture":
    page_architecture()
