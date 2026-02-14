# 🧬 DCGAN Cancer Image Generator

A Deep Convolutional Generative Adversarial Network (DCGAN) for generating synthetic breast cancer histopathology images. Includes a full training pipeline and an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
GAN SKILL PROJECT-3/
├── app.py                    # Streamlit dashboard app
├── configs/
│   ├── data_config.yaml      # Dataset configuration
│   └── train_config.yaml     # Training hyperparameters
├── src/
│   ├── generator.py          # Generator network
│   ├── discriminator.py      # Discriminator network
│   ├── dcgan_model.py        # Combined DCGAN model
│   ├── data_loader.py        # Dataset & DataLoader
│   ├── train_dcgan.py        # Training script
│   └── utils/
│       ├── config.py         # Config loader
│       └── logger.py         # CSV & TensorBoard logger
├── checkpoints/              # Saved model weights (.pt)
├── samples/                  # Generated image grids
├── logs/                     # Training metrics (CSV + TensorBoard)
└── data/                     # Dataset directory
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy pyyaml tensorboard streamlit pandas pillow
```

### 2. Train the Model

```bash
# Quick test with synthetic data
python src/train_dcgan.py --create_sample_data --epochs 10

# Full training on real data
python src/train_dcgan.py --epochs 100 --batch_size 64
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

The app has 4 pages:

| Page | Description |
|------|-------------|
| 📊 **Training Dashboard** | Interactive loss & accuracy charts |
| 🎨 **Image Generator** | Generate new images from trained models |
| 📸 **Sample Browser** | Browse samples across epochs with comparison |
| 🧠 **Model Architecture** | View network details & training config |

---

## ⚙️ Training Options

```bash
python src/train_dcgan.py \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.0002 \
  --latent_dim 100 \
  --device cuda \
  --save_interval 10 \
  --sample_interval 5
```

---

## 📊 Model Architecture

- **Generator**: Latent vector (100-dim) → 64×64 RGB image via transposed convolutions
- **Discriminator**: 64×64 RGB image → real/fake probability via strided convolutions
- **Stabilization**: Label smoothing, gradient clipping, dropout
- **Optimizer**: Adam (lr=0.0002, β1=0.5, β2=0.999)
