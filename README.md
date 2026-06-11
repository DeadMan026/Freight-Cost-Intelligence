# Vendor Invoice Intelligence Portal

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![Great Expectations](https://img.shields.io/badge/Great--Expectations-Data%20Validation-FF6B6B.svg)](https://greatexpectations.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-9cf.svg)](https://shap.readthedocs.io/)

A production-grade, end-to-end Machine Learning ecosystem designed to automate vendor invoice auditing and optimize freight cost forecasting. This platform integrates advanced predictive modeling with rigorous data validation and a Human-in-the-Loop (HITL) framework to ensure financial accuracy and operational efficiency.

---

## 📑 Strategic Overview

The system addresses financial leakage and high manual auditing costs by providing two core intelligence services:
1.  **Risk Detection:** Identifying abnormal or suspicious invoices using a Random Forest classifier.
2.  **Cost Estimation:** Forecasting expected freight charges using robust regression techniques.

By combining automated scoring with expert human feedback, the portal continuously adapts to changing vendor behaviors and delivery patterns.

---

## ✨ Core Capabilities

### 🛡️ Data Integrity & Quality Assurance
We employ a "Validation-First" architecture to maintain high model reliability:
*   **Schema Enforcement:** Automated checks using **Great Expectations (GX)** to verify data types, ranges, and completeness.
*   **Inference Guardrails:** Real-time validation of incoming invoice data to prevent erroneous predictions and handle outliers gracefully.
*   **Distributional Monitoring:** Post-scaling validation to ensure feature stability before they are processed by the machine learning models.

### 🔍 Intelligent Auditing & Explainability
*   **Automated Flagging:** Real-time risk scoring for single invoices or batch CSV uploads.
*   **Model Transparency (SHAP):** Every flagged decision is accompanied by a local explanation chart, highlighting the specific features (e.g., freight cost spikes, receiving delays) that contributed to the risk score.
*   **Batch Processing:** Scalable processing of bulk invoice data with immediate CSV export of risk assessments.

### 🔄 Human-in-the-Loop (HITL) Continuous Learning
The platform bridges the gap between AI and human expertise:
*   **Expert Feedback Loop:** Auditors can confirm or correct model decisions directly within the UI.
*   **Weighted Retraining:** Human-validated samples are prioritized during model retraining (carrying 5x higher weight), ensuring the system aligns with expert institutional knowledge.
*   **Automated Model Evolution:** One-click retraining workflows that handle data merging, weighting, and hot-deployment of updated models.

---

## 🛠 Technical Architecture

| Layer | Component |
| :--- | :--- |
| **User Interface** | Streamlit-based Interactive Dashboard |
| **Data Validation** | Great Expectations (GX) for multi-stage quality checks |
| **Predictive Core** | Scikit-Learn (Random Forest, Linear Regression) |
| **Explainability** | SHAP (Shapley Additive Explanations) |
| **Data Storage** | SQLite for raw ingestion, feedback capture, and audit trails |
| **Preprocessing** | Robust scaling and heuristic feature engineering |

---

## 📂 Project Structure

*   `app.py`: Central portal entry point and UI logic.
*   `data_validation.py`: Core GX validation engine for training and inference.
*   📁 `Inference/`: Modular prediction scripts for classification and regression.
*   📁 `Invoice_Flagging/`: Training pipeline for risk detection with HITL integration.
*   📁 `Freight_Cost_Prediction/`: Regression pipeline for shipping cost forecasting.
*   📁 `models/`: Versioned model artifacts, scalers, and explanation backgrounds.
*   📁 `Scripts/`: Database ingestion and utility scripts.

---

## 🚀 Getting Started

1.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Database Initialization:**
    ```bash
    python Scripts/ingestion_db.py
    ```

3.  **Run Application:**
    ```bash
    streamlit run app.py
    ```
