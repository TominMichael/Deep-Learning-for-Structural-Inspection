# Autonomous Drone-Based System for High-Precision Structural Crack Detection


*A state-of-the-art deep learning pipeline for identifying, measuring, and analyzing structural defects on high-rise infrastructure, now enhanced with Gemini AI.*

---

## 🚀 Project Overview

This repository contains the complete codebase for an industrial-grade, deep learning system designed for the autonomous inspection of structural cracks on buildings using drones. The primary objective is to move beyond simple detection and provide a quantitative, actionable analysis of defects, focusing on cracks with a width of **3mm or greater**.

The system is built to be robust, reliable, and data-driven, leveraging a state-of-the-art **U-Net++ architecture with an EfficientNet backbone** to perform precise semantic segmentation. This allows for the accurate measurement of critical crack properties such as **length, width, and orientation**.

To further enhance the project's analytical power, this system now integrates the **Gemini API** to provide generative AI insights, including automated report summaries and interactive analysis of new data.

This project was meticulously developed by **Ashish480** with a focus on professional MLOps practices, from data validation to model training and AI-powered evaluation.

## ✨ Key Features

-   **High-Precision Segmentation:** Utilizes a U-Net++ model to generate pixel-perfect masks of detected cracks.
-   **Quantitative Analysis:** Capable of measuring key geometric properties (length, width, orientation) from predicted masks.
-   **Industrial-Grade Filtering:** Designed to focus on significant defects (e.g., >3mm width) and ignore superficial blemishes.
-   **Robust Data Pipeline:** Built on a perfectly balanced and validated dataset of over 27,000 images, addressing challenges of class imbalance and inconsistent resolutions.
-   **Advanced Training Techniques:** Employs transfer learning, extensive data augmentation, **Focal Tversky Loss** to handle extreme pixel imbalance, learning rate scheduling, and early stopping.
-   **AI-Powered Insights (New!):** Integrates the **Gemini API** to automatically generate reports, explain complex code, and perform real-time analysis on user-uploaded images.
-   **Modular Codebase:** The project is structured with professional best practices for easy maintenance and scalability.

## 🛠️ Model Architecture

The core of this project is a state-of-the-art semantic segmentation model chosen for its precision.

-   **Model:** **U-Net++**
-   **Backbone:** **EfficientNet** (configurable, e.g., B1, B4, B5)
-   **Pre-trained Weights:** Utilizes **ImageNet** pre-trained weights for the encoder (Transfer Learning).
-   **Final Activation:** **Sigmoid**

This architecture was specifically chosen because the dense skip connections in U-Net++ are exceptionally good at preserving the fine-grained boundary details critical for accurate measurement.

## 📂 Project Structure

crack_detection_project/
|
├── data/
│   └── final_dataset_ready_for_training/
│       ├── train/
│       ├── validation/
│       ├── test/
│       └── metadata.csv
├── models/
│   └── unetplusplus_efficientnet-b5_best.pth
├── logs/
│   └── training_history.csv
└── src/
└── ... (modular Python code)


## ⚙️ Setup and Installation

To set up the environment and run this project, please follow these steps.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Set Up the Python Environment
This project uses a dedicated Python environment. Choose one of the following methods.

Method A: Using venv (Lightweight)

Bash

# Create a virtual environment
python -m venv venv

# Activate the environment (on Windows)
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install pandas opencv-python-headless albumentations matplotlib seaborn tqdm scikit-learn jupyterlab ipykernel segmentation-models-pytorch
Method B: Using conda

Bash

# Create the conda environment
conda create -n crack_detection python=3.9 -y

# Activate the environment
conda activate crack_detection

# Install PyTorch with GPU support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
conda install -c conda-forge pandas opencv albumentations matplotlib seaborn tqdm scikit-learn jupyterlab ipywidgets -y
pip install segmentation-models-pytorch "networkx<3.0"
3. Register Kernel with Jupyter
This makes your environment available directly within Jupyter.

Bash

python -m ipykernel install --user --name=crack_detection_env --display-name="Python (Crack Detection)"
▶️ How to Run
1. Configure the Training
All paths and hyperparameters can be modified in the config.py file or at the top of the main Jupyter Notebook.

2. Run the Training
Execute the main training script or notebook.

Bash

# To run the Python script from the command line:
python src/crack_segmentation/main.py

# Alternatively, run the cells within the provided Jupyter Notebook.
The script will automatically detect your hardware, load the data, build the model, and begin the training process. It saves the best-performing model to the models/ directory.

3. Launch the Interactive Web App
The project includes a single-page web application (index.html) to showcase the results and interact with the Gemini API. Simply open this file in a web browser.

🤖 AI-Powered Insights with Gemini
This project is enhanced with generative AI capabilities through the Gemini API:

Automated Reporting: Generates high-level narrative summaries of the model's performance metrics, translating complex data into understandable insights.

Code Explanation: Provides line-by-line explanations of complex code sections, such as the FocalTverskyLoss function, making the project's technical aspects more accessible.

Interactive Image Analysis: Allows users to upload a new image of a crack and receive a real-time, qualitative analysis of its potential severity from the Gemini multimodal model.

📊 Results
After a full training run, the model achieves excellent performance on the unseen test set, with a Dice Score of over 0.85. The learning curves show a healthy training progression, indicating a well-generalized model.

(Here you can insert your final learning curve plots and prediction images)

Example of model predictions on unseen test images, with the predicted boundary overlaid in green.

🚀 Future Work
Hyperparameter Optimization: Systematically tune learning rates and loss function parameters to further boost performance.

Larger Backbones: Train with larger backbones like efficientnet-b7 on more powerful hardware for higher accuracy.

Deployment: Integrate the trained model into a real-time inference pipeline on a drone's companion computer.

3D Model Integration: Project the 2D crack detections onto a 3D model of the building (a "Digital Twin") for precise geospatial localization.
