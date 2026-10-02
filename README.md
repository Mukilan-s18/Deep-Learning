<div align="center">

# 🧠 Deep Learning Laboratory Experiments & Mini Project
### End-to-End Implementations with PyTorch 2.x • Accelerated on Apple Silicon Metal Performance Shaders (MPS)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Apple MPS GPU](https://img.shields.io/badge/Hardware-Apple%20MPS%20GPU-000000?style=for-the-badge&logo=apple&logoColor=white)](https://developer.apple.com/metal/pytorch/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Course: REC AIML](https://img.shields.io/badge/Coursework-Deep%20Learning%20Lab-purple?style=for-the-badge)](https://rajalakshmi.org/)

<p align="center">
  <b>A comprehensive, production-grade collection of 11 deep learning experiments and a capstone mini project.</b><br>
  Covering multi-layer perceptrons, convolutional neural networks, sequential models (RNN/LSTM), generative models (VAE/GAN), word embeddings, and unsupervised aerospace anomaly detection - strictly trained on real-world datasets with Apple Silicon GPU acceleration.
</p>

[Overview](#-overview) •
[Experiments Table](#-experiments-overview) •
[Mini Project Spotlight](#-mini-project-spotlight) •
[Repository Structure](#-repository-structure) •
[Hardware Acceleration](#-hardware-acceleration--apple-mps) •
[Quickstart](#-quickstart--usage) •
[Author](#-author)

</div>

---

## 📖 Overview

This repository contains the complete laboratory coursework and capstone mini project for the **Deep Learning Laboratory** in the Department of Artificial Intelligence and Machine Learning. 

### 🌟 Key Tenets & Engineering Highlights:
- **100% Real-World Data Sources:** Zero reliance on synthetic random arrays. Every model is trained on real-world clinical, vision, sequential, textual, or aerospace telemetry datasets sourced from scikit-learn built-ins, standard benchmarks, and the UCI Machine Learning Repository.
- **Hardware Acceleration:** Native acceleration via Apple Silicon **Metal Performance Shaders (`mps`)**, providing unified memory benefits, rapid mini-batch streaming, and high compute throughput.
- **Systematic Evaluation:** Every experiment generates quantitative diagnostic metrics, convergence curves, and high-resolution publication-quality visual plots.
- **End-to-End Reproducibility:** Each experiment is organized as an independent, self-contained directory containing both the executed Jupyter Notebook (`experiment.ipynb`) and its output figures.

---

## 🛠️ Tools & Technologies

| Layer | Technologies / Frameworks | Purpose |
| :--- | :--- | :--- |
| **Deep Learning Framework** | **PyTorch 2.x** (`torch`, `torch.nn`, `torch.optim`) | Model definition, backpropagation, and tensor computation |
| **Hardware Acceleration** | **Apple Silicon MPS (Metal Performance Shaders)** | GPU-accelerated tensor training on macOS ARM64 |
| **Machine Learning & Preprocessing**| **Scikit-Learn** (`StandardScaler`, `train_test_split`, `metrics`) | Normalization, dataset splitting, ROC-AUC, classification reports |
| **Data Ingestion & Handling** | **OpenML / Pandas / NumPy** | Fetching UCI datasets, array operations, and DataFrame manipulation |
| **Visualization & Plots** | **Matplotlib & Seaborn** | Training loss curves, confusion matrices, ROC curves, PCA/t-SNE projections |
| **Interactive Environments** | **JupyterLab / IPython Notebooks** | Interactive narrative, step-by-step model execution, and visual inspection |

---

## 🧪 Experiments Overview

| # | Experiment | Architecture / Algorithm | Dataset | Domain / Task | Artifacts |
|:---:|:---|:---|:---|:---|:---:|
| **01** | [**Neural Network from Scratch**](Experiment_01_Neural_Network_From_Scratch/) | 2-Layer Neural Network (NumPy manual forward/backward passes, Cross-Entropy) | **Optical Recognition of Handwritten Digits** (1,797 samples, 64 features) | Computer Vision / Multi-Class Classification | [`digits_loss_plot.png`](Experiment_01_Neural_Network_From_Scratch/digits_loss_plot.png) |
| **02** | [**MLP for Classification**](Experiment_02_MLP_Iris/) | Multi-Layer Perceptron (PyTorch `nn.Linear`, ReLU, Adam) | **Fisher's Iris Dataset** (150 samples, 4 morphological features) | Biology / Botanical Classification | [`iris_mlp_loss.png`](Experiment_02_MLP_Iris/iris_mlp_loss.png) |
| **03** | [**SGD vs. Adam Optimization**](Experiment_03_SGD_vs_Adam/) | Deep MLP comparative optimization (SGD with Momentum vs. Adaptive Adam) | **Wisconsin Breast Cancer Diagnostic** (569 samples, 30 features) | Oncology / Optimizer Dynamics & Convergence | [`optimizer_comparison.png`](Experiment_03_SGD_vs_Adam/optimizer_comparison.png) |
| **04** | [**CNN for Image Classification**](Experiment_04_CNN_Digits/) | Convolutional Neural Network (`Conv2d`, `MaxPool2d`, `Flatten`, `Linear`) | **Handwritten Digits** (8×8 pixel matrices) | Computer Vision / Feature Map Extraction | [`digits_cnn_filters.png`](Experiment_04_CNN_Digits/digits_cnn_filters.png) |
| **05** | [**Autoencoder for Dimensionality Reduction**](Experiment_05_Autoencoder/) | Symmetric Bottleneck Autoencoder (13 $\to$ 2 $\to$ 13 with MSE reconstruction) | **UCI Wine Dataset** (178 samples, 13 chemical constituents) | Chemistry / Non-Linear Latent Projection | [`wine_latent.png`](Experiment_05_Autoencoder/wine_latent.png) |
| **06** | [**RNN for Sequence Processing**](Experiment_06_RNN_Sequence/) | Recurrent Neural Network (`nn.RNN`, Hidden states across lookback windows) | **International Airline Passengers** (Monthly time-series 1949-1960) | Time-Series / Sequential Trend Forecasting | [`rnn_loss.png`](Experiment_06_RNN_Sequence/rnn_loss.png) |
| **07** | [**LSTM Sequential Architecture**](Experiment_07_LSTM/) | Long Short-Term Memory Network (`nn.LSTM`, Input/Forget/Output gating) | **Daily Total Female Births in California** (1959 time-series) | Healthcare / Long-Range Sequence Modeling | [`lstm_predictions.png`](Experiment_07_LSTM/lstm_predictions.png) |
| **08** | [**Variational Autoencoder (VAE)**](Experiment_08_Variational_Autoencoder/) | Probabilistic VAE (Reparameterization Trick, Latent $\mu$ and $\log\sigma^2$, KL Divergence) | **Handwritten Digits** (8×8 pixel matrices) | Generative AI / Continuous Manifold Sampling | [`digits_vae_generated.png`](Experiment_08_Variational_Autoencoder/digits_vae_generated.png) |
| **09** | [**Word2Vec Neural Embedding**](Experiment_09_Word2Vec_Embedding/) | Skip-gram Architecture (Target-to-Context prediction, Latent Embedding matrix) | **Complete Shakespeare Corpus** (Real textual dialogue dataset) | NLP / Semantic Word Vector Spaces | [`word_embeddings.png`](Experiment_09_Word2Vec_Embedding/word_embeddings.png) |
| **10** | [**Generative Adversarial Network (GAN)**](Experiment_10_GAN_Digits/) | Minimax Deep GAN (Generator $G(z)$ vs. Discriminator $D(x)$, BCE loss) | **Handwritten Digits** (Noise vector $z \sim \mathcal{N}(0, I)$) | Generative Modeling / Adversarial Synthesis | [`digits_gan_output.png`](Experiment_10_GAN_Digits/digits_gan_output.png) |
| **11** | [**Capstone Mini Project**](Experiment_11_Mini_Project/) | Semi-Supervised Baseline ($9 \to 3 \to 9$) vs. Deep Regularized Autoencoder ($9 \to 32 \to 16 \to 4 \to 16 \to 32 \to 9$) | **NASA Statlog Shuttle Flight Telemetry** (UCI ID 148: 58,000 samples, 9 sensor channels) | Aerospace Engineering / Mission Anomaly Detection | [`mini_project_results.png`](Experiment_11_Mini_Project/mini_project_results.png) |

---

## 🛰️ Mini Project Spotlight

### **Autonomous Spacecraft Telemetry Anomaly Detection in NASA Space Shuttle Missions**
> **Folder:** [`Experiment_11_Mini_Project/`](Experiment_11_Mini_Project/)  
> **Reports:** 📕 [**PDF Report (51 Pages)**](Experiment_11_Mini_Project/Mini_Project_Report.pdf) • 📘 [**Word (.docx)**](Experiment_11_Mini_Project/Mini_Project_Report.docx) • 📗 [**Academic Markdown**](Experiment_11_Mini_Project/MINI_PROJECT_REPORT.md) • 📙 [**Jupyter Notebook**](Experiment_11_Mini_Project/experiment.ipynb)

In crewed aerospace operations, ground-control telemetry monitoring faces transmission latency and plasma communication blackouts during atmospheric re-entry. Because spaceflight failure modes cannot be simulated in advance, this project implements **unsupervised semi-supervised anomaly detection** trained exclusively on nominal flight operations (Class 1: Radiator Flow). Any unseen subsystem failure produces an abnormal **Reconstruction Error ($\text{MSE}$)** that triggers real-time autonomous alerts.

<div align="center">

| Metric | Baseline Linear Autoencoder ($9 \to 3 \to 9$) | Advanced Deep Autoencoder ($9 \to 32 \to 16 \to 4 \to 16 \to 32 \to 9$) |
| :--- | :---: | :---: |
| **Trainable Parameters** | **69** | **2,125** |
| **Operational Decision Threshold ($\tau$, 95th %ile)** | **0.0602** | **0.1047** |
| **ROC-AUC Score** | **0.9832** | **0.9627** |
| **Classification Accuracy** | **93.98%** | **91.08%** |
| **Precision (Positive Predictive Value)** | **83.09%** | **80.68%** |
| **Recall / Sensitivity (True Positive Rate)** | **90.22%** | **76.71%** |
| **Specificity (True Negative Rate)** | **95.00%** | **95.00%** |
| **F1-Score** | **0.8651** | **0.7865** |
| **Anomalies Caught / Total** | **11,200 / 12,414** | **9,523 / 12,414** |
| **NASA Subtype Recall** | **100% on High Pressure • 98.8% on Low Flow • 92.3% on Critical Failures** |

<br/>

| Reconstruction Error Separation | Receiver Operating Characteristic (ROC) |
| :---: | :---: |
| ![Reconstruction Error](Experiment_11_Mini_Project/mini_project_results.png) | ![ROC Curves](Experiment_11_Mini_Project/roc_curves_comparison.png) |

</div>

---

## ⚡ Hardware Acceleration - Apple MPS

All experiments are optimized for execution on **Apple Silicon (M1/M2/M3/M4 Series)** using PyTorch's native Metal Performance Shaders (`mps`) backend. This delivers accelerated matrix multiplications directly within the unified memory architecture.

```python
import torch

# Automatic device selection: Apple MPS GPU with graceful CPU fallback
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Active Compute Device: {device}")

# Move model and data tensors to unified GPU memory
model = model.to(device)
inputs = inputs.to(device)
```

---

## 📁 Repository Structure

```
Deep-Learning/
├── Experiment_01_Neural_Network_From_Scratch/
│   ├── experiment.ipynb                 # NumPy 2-layer NN implementation
│   └── digits_loss_plot.png             # Training loss curve
├── Experiment_02_MLP_Iris/
│   ├── experiment.ipynb                 # PyTorch MLP on Iris
│   └── iris_mlp_loss.png                # Multi-class cross-entropy loss
├── Experiment_03_SGD_vs_Adam/
│   ├── experiment.ipynb                 # SGD vs Adam optimizer benchmark
│   └── optimizer_comparison.png         # Comparative convergence curves
├── Experiment_04_CNN_Digits/
│   ├── experiment.ipynb                 # Conv2d digit classifier
│   └── digits_cnn_filters.png           # Learned convolutional feature filters
├── Experiment_05_Autoencoder/
│   ├── experiment.ipynb                 # Non-linear autoencoder dimensionality reduction
│   └── wine_latent.png                  # 2D latent bottleneck projection
├── Experiment_06_RNN_Sequence/
│   ├── experiment.ipynb                 # PyTorch nn.RNN sequential forecaster
│   └── rnn_loss.png                     # Time-series MSE loss convergence
├── Experiment_07_LSTM/
│   ├── experiment.ipynb                 # Long Short-Term Memory sequence model
│   └── lstm_predictions.png             # Actual vs predicted sequence plot
├── Experiment_08_Variational_Autoencoder/
│   ├── experiment.ipynb                 # Probabilistic VAE with reparameterization
│   └── digits_vae_generated.png         # Novel synthetic generated digits
├── Experiment_09_Word2Vec_Embedding/
│   ├── experiment.ipynb                 # Skip-gram word embedding on Shakespeare
│   └── word_embeddings.png              # 2D semantic embedding vector space
├── Experiment_10_GAN_Digits/
│   ├── experiment.ipynb                 # Minimax GAN Generator vs Discriminator
│   └── digits_gan_output.png            # Generated digit images across training
├── Experiment_11_Mini_Project/
│   ├── README.md                        # Full coursework project report
│   ├── experiment.ipynb                 # Executed PyTorch Mini-Batch DataLoader notebook
│   ├── metrics.json                     # Computed performance benchmarks
│   ├── mini_project_results.png         # Reconstruction error distribution (log scale)
│   ├── training_loss_curves.png         # Training & validation MSE convergence
│   ├── roc_curves_comparison.png        # ROC curves (Baseline 0.9832 vs Deep 0.9627)
│   ├── confusion_matrices.png           # Confusion matrix heatmaps
│   ├── latent_space_pca.png             # 2D PCA projection of flight telemetry
│   └── anomaly_subtypes_detection.png   # Per-subtype recall across 6 failure modes
├── .gitignore                           # Git ignore rules
└── README.md                            # Main repository documentation (this file)
```

---

## 🚀 Quickstart & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Mukilan-s18/Deep-Learning.git
cd Deep-Learning
```

### 2. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision scikit-learn pandas numpy matplotlib
```

### 3. Launch JupyterLab
```bash
jupyter lab
```
Navigate to any experiment directory (e.g., `Experiment_11_Mini_Project/experiment.ipynb`) and run all cells interactively.

---

## 👤 Author

| Field | Details |
| :--- | :--- |
| **Student Name** | Mukilan S |
| **Roll Number** | `241501123` |
| **Department** | Artificial Intelligence & Machine Learning (AI & ML) |
| **Institution** | Rajalakshmi Engineering College (REC) |
| **GitHub Profile** | [@Mukilan-s18](https://github.com/Mukilan-s18) |
| **Repository** | [Mukilan-s18/Deep-Learning](https://github.com/Mukilan-s18/Deep-Learning) |

---

## 📄 License

This repository is published for **academic and educational purposes** as part of the Deep Learning Laboratory curriculum. All code and implementations adhere to academic integrity guidelines.
