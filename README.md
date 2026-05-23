# Vendor Invoice Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E.svg)](https://scikit-learn.org/)

An end-to-end machine learning solution designed to streamline finance operations by predicting freight costs and identifying high-risk vendor invoices. This system transforms raw transactional data into actionable insights, reducing manual oversight and preventing financial leakage.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Data Source](#data-source)
- [EDA & Insights](#eda--insights)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Application](#application)
- [Project Structure](#project-structure)
- [How to Run This Project](#how-to-run-this-project)
- [Author and Contact](#author-and-contact)

---

## Project Overview
This project addresses two critical challenges in vendor management: accurately forecasting logistics expenses and auditing invoices for anomalies. By leveraging historical purchase and inventory data, the system provides:
1. **Freight Prediction:** Estimating expected shipping costs based on invoice value.
2. **Risk Flagging:** A classification engine that detects invoices requiring manual intervention due to abnormal patterns in quantity, cost, or delivery delays.

## Business Objective
*   **Freight Cost Prediction:** Automate the estimation of freight expenses to improve budgeting accuracy and provide a baseline for vendor negotiations.
*   **Invoice Risk Flagging:** Protect the bottom line by identifying high-risk invoices (e.g., overcharges, unusual quantities, or delivery inconsistencies) before payment processing.

## Data Source
The system utilizes a relational **SQLite** database (`inventory.db`) containing integrated data from:
*   **Purchases & Purchase Prices:** Historical transactional records and unit costs.
*   **Vendor Invoices:** Direct invoice records including freight and total dollars.
*   **Inventory (Begin/End):** Contextual data on stock levels to correlate with purchasing patterns.

## EDA & Insights
Before modeling, I performed extensive Exploratory Data Analysis to understand the underlying drivers of cost:
*   **Correlation Analysis:** Identified strong linear relationships between invoice value (Dollars) and freight costs.
*   **Outlier Detection:** Spotted abnormal freight-to-total-dollar ratios that helped define the "Risk Flag" logic.
*   **Temporal Patterns:** Analyzed receiving delays to identify vendor performance trends.

## Models Used
### 1. Regression (Freight Prediction)
*   **Baseline:** Linear Regression.
*   **Final Choice:** Linear Regression was selected for its high interpretability and strong performance on the core feature (Invoice Dollars).

### 2. Classification (Invoice Flagging)
*   **Baseline:** Logistic Regression.
*   **Final Model:** **Random Forest Classifier**.
*   **Optimization:** Implemented `GridSearchCV` to tune hyperparameters, specifically optimizing for the **F1-Score** to maintain a balance between precision and recall given the class imbalance in risk flagging.

## Evaluation Metrics
To ensure reliability, models were evaluated using the following metrics:
*   **Regression:** 
    *   **MAE (Mean Absolute Error):** To understand the average dollar deviation.
    *   **RMSE (Root Mean Squared Error):** To penalize larger forecasting errors.
    *   **R-Squared:** Measuring the variance captured by the model.
*   **Classification:**
    *   **F1-Score:** The primary metric for risk detection.
    *   **Precision & Recall:** To minimize false positives while ensuring high-risk invoices aren't missed.
    *   **Accuracy:** Overall prediction correctness.

## Application
The system is deployed as an interactive **Streamlit** web portal. It allows finance teams to:
*   Input invoice details manually for real-time scoring.
*   View instant risk assessments (Safe vs. Manual Approval Required).
*   Get quick freight estimates for cost verification.

---

## Project Structure
```text
.
├── app.py                      # Streamlit Web Application
├── data/                       # Database and raw CSV storage
├── models/                     # Saved Pickle files (Models & Scalers)
├── Notebooks/                  # Detailed EDA and Experimentation
├── Freight_Cost_Prediction/    # Regression Training Pipeline
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
├── Invoice_Flagging/           # Classification Training Pipeline
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   └── train.py
├── Inference/                  # Core Prediction Logic for App
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
└── Scripts/                    # Data Ingestion & DB Scripts
    └── ingestion_db.py
```

---

## How to Run This Project
1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd Invoice_Intelligence
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Data:**
    ```bash
    python Scripts/ingestion_db.py
    ```
4.  **Train Models (Optional - pre-trained models are in /models):**
    ```bash
    python Freight_Cost_Prediction/train.py
    python Invoice_Flagging/train.py
    ```
5.  **Launch the App:**
    ```bash
    streamlit run app.py
    ```

---

## Author and Contact
**Souvik Sinha**
*   **GitHub:** [DeadMan026](https://github.com/DeadMan026)
*   **Email:** 27souviksinha@gmail.com

*(Feel free to reach out for collaboration or inquiries!)*
