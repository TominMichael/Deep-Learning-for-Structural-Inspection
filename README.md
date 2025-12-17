# 🏗️ Deep Learning for Structural Inspection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

*An industrial-grade, autonomous system for high-precision structural crack detection and quantitative analysis using advanced Deep Learning.*

[Features](#-key-features) •
[Architecture](#-system-architecture) •
[Getting Started](#-getting-started) •
[Results](#-performance-analysis)

</div>

---

## 🚀 Project Overview

**Deep Learning for Structural Inspection** is a cutting-edge solution designed to revolutionize how high-rise infrastructure is monitored. Moving beyond simple defect detection, this system provides actionable, quantitative intelligence by identifying and measuring structural cracks with extreme precision.

Built to process high-resolution drone imagery, the pipeline leverages a robust deep learning architecture to segment defects and calculate critical metrics—**length, width, and orientation**—empowering engineers to make data-driven maintenance decisions.

## ✨ Key Features

- **🎯 Precision Segmentation**: Utilizes **U-Net++** with an **EfficientNet-B7** encoder for state-of-the-art pixel-level accuracy.
- **📏 Quantitative Analysis**: Automated geometric analysis pipeline extracts:
  - Crack Length (in pixels/physical units)
  - Crack Width (min/max profile)
  - Orientation (degrees relative to structure)
- **🧠 Advanced Optimization**: Implements **Focal Tversky Loss** to handle severe class imbalance between background and crack pixels.
- **🔄 Robustness**: Extensive data augmentation pipeline using **Albumentations** (Flip, Rotate, Brightness/Contrast, Affine) ensures performance across varying lighting and weather conditions.
- **📊 Detailed Reporting**: Generates comprehensive CSV logs and visualization plots for validation metrics (Dice, IoU, Precision, Recall).

## 🛠️ System Architecture

The core of the system is built on a modular, scalable deep learning pipeline:

| Component | specification |
|-----------|---------------|
| **Framework** | PyTorch |
| **Model Arch** | U-Net++ (Nested U-Net) |
| **Backbone** | EfficientNet-B7 (Pre-trained on ImageNet) |
| **Input Resolution** | 512x512 |
| **Optimizer** | AdamW |
| **Loss Function** | Focal Tversky Loss (Alpha=0.7, Beta=0.3, Gamma=2.0) |

## 📂 Repository Structure

```
├── src/
│   ├── train/
│   │   └── train.py           # Main training training loop with validation
│   ├── geometry analysis/
│   │   └── crack_gemoetryanalysis.ipynb # Geometric properties extraction pipeline
│   ├── test pythoncodes/      # Testing utilities
│   ├── Data Validation/       # Scripts for dataset integrity checks
│   └── organizer py code/     # Data file management helpers
├── results/                   # Output directory for models and logs
└── README.md                  # Project documentation
```

## 💻 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and a CUDA-capable GPU.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TominMichael/Deep-Learning-for-Structural-Inspection.git
   cd Deep-Learning-for-Structural-Inspection
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install torch torchvision pandas numpy opencv-python matplotlib seaborn albumentations tqdm scikit-learn segmentation-models-pytorch
   ```

### Usage

#### Training the Model
To start training the segmentation model, configure the paths in `src/train/train.py` and run:

```bash
python src/train/train.py
```
*Note: Ensure `LOCAL_INPUT_DIR` in the script points to your dataset.*

#### Geometric Analysis
Post-processing analysis is handled via the Jupyter Notebook. Open `src/geometry analysis/crack_gemoetryanalysis.ipynb` to process predicted masks and export geometric data.

## 📈 Performance Analysis

The system logs comprehensive metrics during training and validation:

- **Dice Score**: Primary metric for overlap accuracy.
- **IoU (Intersection over Union)**: Robust measure of segmentation quality.
- **Specificity**: Ensures non-crack areas are correctly identified.

Visualizations of training progress and validation distributions are automatically saved to the `analysis/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
    <b>Developed with ❤️ by <a href="https://github.com/TominMichael">Tomin Michael</a></b>
</div>
