# 🎨 Project 5: Generative Models Showdown

> VAE vs GAN vs Diffusion - Three approaches to image generation

## 🎯 Overview

This project implements three different generative models from scratch using PyTorch:

1. **VAE (Variational Autoencoder)** - Compresses and reconstructs images
2. **GAN (Generative Adversarial Network)** - Generates images via competition
3. **Diffusion Model** - Generates images via gradual denoising

All models are trained on the MNIST handwritten digits dataset.

## 📁 Project Structure
project5_generative_models/
│
├── config.py # Configuration
├── main.py # Entry point
├── requirements.txt # Dependencies
├── README.md # This file
│
├── models/
│ ├── vae.py # VAE implementation
│ ├── gan.py # GAN implementation
│ └── diffusion.py # Diffusion implementation
│
├── data/
│ └── dataset.py # MNIST data loader
│
├── training/
│ ├── train_vae.py # VAE training
│ ├── train_gan.py # GAN training
│ └── train_diffusion.py # Diffusion training
│
└── utils/
    └── visualization.py # Plotting utilities


## 🚀 Quick Start
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models 
python main.py


🧠 Model Architecture
VAE (Variational Autoencoder)
Input (28×28) → Encoder → Latent (20) → Decoder → Output (28×28)

GAN (Generative Adversarial Network)
Noise (100) → Generator → Fake Image ↔ Discriminator → Real/Fake

Diffusion
Noise → Step 1 → Step 2 → ... → Step N → Image

📊 Comparison
Feature	VAE	GAN	Diffusion
Quality	Good (blurry)	Excellent	Excellent
Speed	Fast	Fast	Slow
Training	Stable	Unstable	Very Stable
Latent Space	Yes	No	No
Difficulty	Easy	Hard	Medium

📈 Results
After training:
VAE: Produces recognizable digits with slight blur
GAN: Creates sharp, realistic digits
Diffusion: Generates high-quality, diverse digits

🔑 Key Concepts Learned
Latent Space - Compressed representation of data
Reparameterization Trick - Differentiable sampling for VAE
Adversarial Training - Two networks competing
Denoising Diffusion - Gradual noise removal
Mode Collapse - GAN failure mode
KL Divergence - Distribution comparison

🎯 Why This Matters
Generative models are the foundation of:
DALL-E, Midjourney - Image generation
ChatGPT - Text generation
DeepFakes - Face generation
Drug Discovery - Molecular generation


🎨 GENERATIVE MODELS SHOWDOWN
============================================================

📦 Training VAE...
Epoch 20/20 | Loss: 45.2341

🎭 Training GAN...
Epoch 50/50 | G Loss: 0.8234 | D Loss: 1.4567

🌀 Training Diffusion...
Epoch 20/20 | Loss: 0.0234
