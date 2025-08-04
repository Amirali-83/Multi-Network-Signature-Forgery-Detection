# 🖋️ Multi-Network Signature Forgery Detection

A machine learning project for detecting forged signatures using **Convolutional Neural Networks (CNN)** and **Siamese Networks**.  
The application is built with **TensorFlow** for training and **Flask + HTML/CSS** for deployment.

---

## About the Project
This project detects whether two signature images belong to the **same person (genuine)** or **different people (forgery)**.  
It uses:
- **CNN** for feature extraction
- **Siamese Network** for similarity learning
- **Flask** to serve predictions in a simple web interface

The trained model achieves high accuracy on the CEDAR signature dataset.

---

## 📂 Dataset
- **Name:** CEDAR Signature Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/matteocarnebella/cedar-signatures)
- **Description:** Contains genuine and forged signatures from multiple individuals.

---

## 🧠 Model Architecture
**1. Base CNN:**
- Conv2D → ReLU → MaxPooling  
- Conv2D → ReLU → MaxPooling  
- Dense(128) for feature embedding

**2. Siamese Network:**
- Two identical CNNs process two signatures
- Output embeddings are compared using absolute difference
- Dense layer outputs similarity score (0–1)
