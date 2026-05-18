# Invoice Intelligence: Freight Analytics & Risk Mitigation System

## Project Overview
This repository hosts a sophisticated Machine Learning and Data Engineering pipeline designed to optimize logistics financial operations. The system transforms raw supply chain data into actionable financial intelligence through automated freight cost prediction and intelligent invoice risk assessment.

## Core Pillars

### 1. High-Integrity Data Architecture
To move beyond the limitations of static spreadsheets, I implemented a centralized **SQLite-based Relational Database Management System (RDBMS)**.
*   **Relational Mapping**: Orchestrates complex joins between `Purchases`, `Vendor Invoices`, and `Inventory` tables.
*   **Data Integrity**: Enforces schema consistency and optimized data retrieval for large-scale logistics datasets.
*   **Scalability**: Built to handle high-volume transaction ingestion from diverse enterprise sources.

### 2. Predictive Freight Cost Modeling (Regression)
The system employs a multi-model regression framework to forecast freight expenditures with high precision.
*   **Modeling Suite**: Evaluation of Linear Regression, Decision Trees, and Random Forest models.
*   **Performance Metrics**: Models are strictly validated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
*   **Automated Selection**: The pipeline automatically identifies and serializes the highest-performing model for downstream production use.

### 3. Invoice Risk Intelligence (Classification)
The system features an automated "Audit-by-Exception" layer to identify high-risk transactions.
*   **Objective**: Detect billing discrepancies, abnormal freight charges, and delivery delays.
*   **Technical Approach**: Utilizes supervised learning (Random Forest) to flag invoices requiring manual human review.
*   **Advanced Feature Engineering**: Incorporates vendor reliability metrics, receiving delays, and pricing deviations to detect anomalous patterns.

## Technical Audits & Continuous Improvement
The project undergoes rigorous internal audits to ensure professional machine learning standards.
*   **Data Leakage Prevention**: Refined feature sets to ensure models learn genuine risk patterns rather than simple arithmetic heuristics.
*   **Robust Preprocessing**: Implementation of `RobustScaler` and log transformations to handle the high variance ($4 to $1.6M+) typical in financial data.
*   **Statistical Validation**: Use of Hypothesis Testing (T-Tests) to confirm that flagged features show statistically significant differences from normal transactions.

## Strategic Roadmap
- [x] Migration to Relational SQLite Database Architecture.
- [x] Production-ready Regression Pipeline for Freight Cost Forecasting.
- [x] Development of Random Forest Classification for Invoice Flagging.
- [x] Technical Audit and Leakage Correction for Classification Models.
- [ ] Integration of a BI Dashboard for Executive Financial Visibility.

## Getting Started

### Installation
Dependencies are managed via the root requirements file:
```bash
pip install -r requirements.txt
```

### Usage
The training scripts use relative paths to the database. Ensure you change into the respective directory before execution.

#### 1. Data Ingestion (Initial Setup)
To ingest raw CSV data into the SQLite database:
```bash
python Scripts/ingestion_db.py
```

#### 2. Freight Cost Prediction
To train and evaluate the regression models:
```bash
cd Freight_Cost_Prediction
python train.py
```

#### 3. Invoice Risk Flagging
To train and evaluate the risk classification models:
```bash
cd Invoice_Flagging
python train.py
```

---
*Architected for Financial Oversight and Logistics Excellence.*
