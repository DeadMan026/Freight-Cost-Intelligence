# Invoice Intelligence: Freight Analytics & Risk Mitigation System

An end-to-end ML pipeline for optimizing logistics financial operations through automated freight cost forecasting and intelligent risk assessment.

## 📌 Table of Contents
* [Project Overview](#project-overview)
* [Business Objective](#business-objective)
* [Data Sources & Architecture](#data-sources--architecture)
* [EDA & Modeling](#eda--modeling)
* [Project Structure](#project-structure)
* [Future Work](#future-work)

## 📖 Project Overview
This system transforms fragmented supply chain data into actionable intelligence. It automates the detection of billing anomalies and predicts freight expenditures, reducing manual audit overhead and financial leakage.

## 🎯 Business Objective
* **Cost Prediction:** Accurate forecasting of freight expenses to improve budgeting.
* **Risk Mitigation:** Automated flagging of high-risk invoices (discrepancies > 0.2%).
* **Process Efficiency:** Consolidation of disparate data sources into a unified analytical workflow.

## 📊 Data Sources & Architecture
* **Sources:** Purchases, Vendor Invoices, and Inventory datasets.
* **Database:** SQLite relational schema for managing complex logistics relationships and ensuring data integrity.
* **Consolidation:** Integrated three disparate sources into a single ingestion pipeline (`ingestion_db.py`).

## ⚙️ EDA & Modeling
* **EDA:** Analysis of price distributions ($4 to $1.6M+), vendor reliability, and route-based cost spikes.
* **Models Used:** 
    * **Regression:** Random Forest, Decision Trees, Linear Regression (for cost forecasting).
    * **Classification:** Random Forest (for flagging invoice anomalies).
    * **Explainability:** SHAP-based feature importance for model transparency.
* **Evaluation Metrics:** 
    * **Regression:** MAE, RMSE.
    * **Classification:** Precision, Recall, F1-Score (optimized for detecting "False Safes").

## 📂 Project Structure
```text
Invoice_Intelligence/
├── Freight_Cost_Prediction/   # Regression pipelines (Train/Eval)
├── Invoice_Flagging/          # Classification pipelines (Train/Eval)
├── Inference/                 # Production inference scripts
├── Scripts/                   # Data ingestion & DB management
├── Notebooks/                 # Exploratory Data Analysis (EDA)
├── data/                      # SQLite DB & Raw CSV storage
└── requirements.txt           # Environment dependencies
```

## 🚀 Future Work
* **Production Inference:** Finalize `inference.py` using `joblib` for real-time single-record and batch CSV processing.
* **Streamlit Dashboard:** Deploy an interactive UI for:
    * **Single Scoring:** Form-based entry for immediate predictions.
    * **Batch Auditing:** CSV upload for bulk risk flagging.
* **Human-in-the-Loop (HITL):** Implement an audit feedback interface where manual "Gold Labels" are captured to retrain and adapt models to new billing patterns.

---
*Architected for Financial Oversight and Logistics Excellence.*
