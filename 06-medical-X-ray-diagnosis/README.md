# 🏥 Project 6: Medical Chest X-ray Diagnosis & Report Generation

> AI-powered chest X-ray analysis with automated clinical report generation

## 🎯 Overview

This project implements a complete medical AI system that:
1. **Analyzes chest X-rays** to detect 14 different diseases
2. **Generates clinical reports** describing findings and impressions

## 🧠 Architecture
X-ray Image → CNN Encoder → Disease Classification (14 diseases)
↓
LSTM Decoder → Clinical Report Generation


## 🚀 Quick Start
pip install -r requirements.txt
python main.py

📊 14 Diseases Detected
Atelectasis
Cardiomegaly
Effusion
Infiltration
Mass
Nodule
Pneumonia
Pneumothorax
Consolidation
Edema
Emphysema
Fibrosis
Pleural_Thickening
Hernia

🏗️ Project Structure
06-medical-X-ray-diagnosis/
│
├── config.py                    # Configuration
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Chest X-ray dataset
│   └── preprocessing.py        # Image preprocessing
│
├── models/
│   ├── __init__.py
│   ├── classifier.py           # Disease classification (CNN)
│   ├── report_generator.py     # Report generation (CNN+LSTM)
│   ├── encoder.py              # Image encoder
│   └── decoder.py              # Text decoder with attention
│
├── training/
│   ├── __init__.py
│   ├── train_classifier.py     # Train disease classifier
│   ├── train_report.py         # Train report generator
│   └── evaluate.py             # Evaluation metrics
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py        # Plotting utilities
│   ├── metrics.py              # Medical metrics
│   └── text_utils.py           # Text processing
│
├── notebooks/
│   └── exploration.ipynb       # Data exploration
│
└── outputs/
    ├── classifications/
    ├── reports/
    └── visualizations/

🔑 Key Concepts
Multi-label Classification: Detect multiple diseases simultaneously
Transfer Learning: Use pre-trained DenseNet for medical images
Attention Mechanism: Focus on relevant image regions
Encoder-Decoder: Convert images to text
Medical Report Generation: Clinical text generation

📚 Dependencies
PyTorch
torchvision
scikit-learn
matplotlib

Built for learning medical AI applications 🏥