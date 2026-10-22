# 🛡️ Motor Insurance Claim Fraud Detection
### An Isolation Forest Anomaly Detector-Based Approach to Insurance Claim Fraud

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

- **Author / Candidate:** BOBBURI PRAVEEN KUMAR
- **Supervisor / Guide:** Dr. P. Neeraja
- **Domain:** Machine Learning & Predictive Analytics

---

## 📌 Abstract
This project proposes an Isolation Forest anomaly detector to surface suspicious motor-insurance claims, addressing the problem of insurance claim fraud through a data-driven approach. The pipeline draws on policy and claims history data, applying suitable pre-processing before training and testing the model on held-out data. The project culminates in an investigator triage queue, bridging the gap between the underlying model and everyday use.

---

## 🚀 Key Results on Held-Out Test Data

- **ROC-AUC Score:** `0.9827`
- **PR-AUC (Average Precision):** `0.9295`
- **Overall Accuracy:** `96.70%`
- **Fraud Detection Recall:** `87.50%`
- **Fraud Triage Precision:** `85.37%`
- **Top 20% Queue Recall:** `96.67%` (Investigating the top 20% highest-risk claims intercepts 96.67% of all fraud)

---

## 🖥️ Application Features
1. **🚨 Investigator Triage Queue:** Real-time priority ranking (HIGH / MEDIUM / LOW), fraud exposure tracking, and one-click actions (*Approve*, *Escalate to SIU*, *Reject*).
2. **🔍 Live Claim Risk Analyzer:** Interactive form to simulate and score incoming motor insurance claims with instant feature-level explainability.
3. **📊 Model Analytics & Mathematical Foundations:** Full interactive charts of ROC and Precision-Recall curves, alongside BST average path length derivations.
4. **📁 Batch File Processing:** Ingest new CSV batches of claims and export a prioritized triage spreadsheet.

---

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/insurance-claim-fraud-detection.git
   cd insurance-claim-fraud-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model training pipeline (optional, pre-trained models already included):**
   ```bash
   python train.py
   ```

4. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Fork or push this repository to your **GitHub** account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"**.
4. Select your repository: `insurance-claim-fraud-detection`.
5. Set Main file path: `app.py`.
6. Click **"Deploy!"** — your app will be live with a shareable public URL in 2 minutes!

---

## 📚 Project Documentation
- [Project Synopsis](docs/PROJECT_SYNOPSIS.md)
- [Comprehensive Academic Report](docs/PROJECT_REPORT.md)
- [Viva Voce Technical Q&A](docs/VIVA_VOICE_QA.md)
