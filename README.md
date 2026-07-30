# FreightAudit — Freight Analytics & Risk Mitigation System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg)](https://www.sqlite.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-9cf.svg)](https://shap.readthedocs.io/)

A full-stack, end-to-end Machine Learning pipeline and interactive application designed to automate the auditing of vendor invoices and predict freight costs. This project leverages **regression** to estimate shipping expenses, **classification** to flag risky invoices, and an active **Human-in-the-Loop (HITL) Continuous Learning** framework to improve models over time.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features & Components](#-key-features--components)
3. [Tech Stack & Methodologies](#-tech-stack--methodologies)
4. [Human-in-the-Loop (HITL) & Retraining Pipeline](#-human-in-the-loop-hitl--retraining-pipeline)
5. [Directory Structure](#-directory-structure)
6. [Getting Started](#-getting-started)

---

## 🚀 Project Overview

**Goal:** Reduce financial leakage and manual auditing workload by predicting expected freight costs and automatically flagging abnormal vendor invoices for manual review.

The system is built for **recruiters and engineering managers** to evaluate a complete ML lifecycle: from raw CSV ingestion into a relational database (SQLite), to exploratory data analysis (Jupyter), feature engineering and scaling (`RobustScaler`), model training (Scikit-Learn), explainability (SHAP), and finally a user-facing deployment (Streamlit).

---

## ✨ Key Features & Components

### 1. Single Invoice Auditing
* Users can input a single invoice's parameters via the Streamlit UI.
* The system evaluates the invoice and outputs a **Probability Risk Score (%)** and a binary Safe/Flagged decision.
* Includes **Model Transparency (Explainable AI)**: Uses **SHAP (SHapley Additive exPlanations)** to generate visual bar charts showing exactly *why* a specific invoice was flagged (e.g., highlighting that freight costs pushed the risk score up).

### 2. Batch Processing & CSV Download
* Upload a CSV containing hundreds of invoices.
* The system scores every row, marking incomplete rows as "Skipped".
* Results can be downloaded immediately as a new CSV via the web app.

### 3. Freight Cost Prediction
* A regression module that forecasts expected freight costs based on the monetary value of the invoice, aiding in budgeting and spotting extreme overcharges.

---

## 🛠 Tech Stack & Methodologies

| Category | Technologies / Details |
| :--- | :--- |
| **Language & UI** | Python, Streamlit, Matplotlib, Seaborn |
| **Database** | SQLite (used for raw data ingestion and auditor feedback storage) |
| **Data Processing** | Pandas, NumPy, Scikit-Learn (`RobustScaler`, `train_test_split`) |
| **Models (Classification)** | Random Forest Classifier (optimized via `GridSearchCV`) |
| **Models (Regression)** | Linear Regression, Decision Trees, Random Forest Regressor |
| **Evaluation Metrics** | Classification: Accuracy, Precision, Recall, **F1-Score** <br> Regression: MAE, RMSE, R² |
| **Explainable AI** | SHAP (TreeExplainer) |

---

## 🔄 Human-in-the-Loop (HITL) & Retraining Pipeline

This project implements a continuous learning system, meaning the model gets smarter as human auditors interact with it.

### How the Feedback Loop Works:
1. **Auditor Review:** When scoring a single invoice, the auditor is presented with the ML model's decision. They can click **"✅ Confirm Correct"** or **"❌ Mark as Error"**.
2. **Database Storage:** This feedback is saved to a SQLite table (`auditor_feedback`), recording the invoice parameters, the auditor's final decision, a timestamp, and the **PO Number**.
3. **The Role of the PO Number:** The `PONumber` acts as a unique identifier. During retraining, the pipeline checks the feedback table. If a human auditor evaluated a specific `PONumber`, the system **deletes the original, rule-based label** for that invoice to prevent duplicate, conflicting records in the training set.
4. **Trigger Retraining:** Directly from the Streamlit UI ("Model Operations" tab), an admin can press a button to retrain the Random Forest model.
5. **Weighted Priority:** The pipeline merges the historical dataset with the newly collected human feedback. The training algorithm is fed these samples using `sample_weights`, where **human decisions carry 5x more weight** than automated rules.
6. **Live Update:** The model is backed up, retrained, saved, and instantly deployed to the live app.

---

## 📂 Directory Structure

| Directory / File | Description |
| :--- | :--- |
| `app.py` | Main Streamlit application entry point. |
| `requirements.txt` | Python dependencies. |
| `sample_invoice.csv` | Sample data for testing the Batch CSV upload feature. |
| 📁 `data/` | Contains the SQLite database (`inventory.db`) and raw `CSVs/`. |
| 📁 `models/` | Serialized `.pkl` files: trained models, scalers, and SHAP background data. Includes automated backups. |
| 📁 `Freight_Cost_Prediction/` | Data preprocessing, evaluation, and training scripts for the regression pipeline. |
| 📁 `Invoice_Flagging/` | Data preprocessing, evaluation, and training scripts for the classification/risk pipeline. |
| 📁 `Inference/` | Lightweight scoring scripts that load `.pkl` models to serve predictions in `app.py`. |
| 📁 `Scripts/` | `ingestion_db.py` to convert raw CSVs into the SQLite relational database. |
| 📁 `Notebooks/` | Jupyter Notebooks used for EDA and initial model prototyping. |

---

## 💻 Getting Started

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd Invoice_Intelligence
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the Database:**
   (Ensure raw CSVs are located in `data/CSVs/`)
   ```bash
   python Scripts/ingestion_db.py
   ```

4. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```
