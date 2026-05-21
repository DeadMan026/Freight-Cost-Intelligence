# Invoice Intelligence: Freight Analytics & Risk Mitigation System

An end-to-end ML pipeline for freight cost forecasting and invoice risk flagging using logistics and procurement data stored in SQLite.

## Table of Contents
- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Data Sources & Architecture](#data-sources--architecture)
- [EDA & Modeling](#eda--modeling)
- [Project Structure](#project-structure)
- [Inference](#inference)
- [Future Work](#future-work)

## Project Overview
This project combines two machine learning workflows:
- freight cost prediction for budgeting and cost analysis
- invoice risk flagging for identifying potentially abnormal vendor invoices

The pipelines are trained from the same consolidated SQLite database built from purchase, invoice, and inventory records.

## Business Objective
- **Cost Prediction:** Forecast freight expenses from operational purchase features.
- **Risk Mitigation:** Flag invoices as risky when they show a material invoice-to-item total mismatch or abnormal receiving delay.
- **Process Efficiency:** Reduce manual review effort by centralizing data ingestion, feature creation, training, and inference.

## Data Sources & Architecture
- **Sources:** `purchases`, `vendor_invoice`, and inventory-related CSV inputs.
- **Database:** SQLite database stored at `data/inventory.db`.
- **Ingestion:** Source CSVs are loaded into the database through [Scripts/ingestion_db.py](/C:/WORKS/PROJECTS/Invoice_Intelligence/Scripts/ingestion_db.py:1).

## EDA & Modeling
- **Notebooks:** Exploration and feature validation live in `Notebooks/`.
- **Freight Cost Modeling:** Training code lives in `Freight_Cost_Prediction/`.
- **Invoice Flagging:** Training code lives in `Invoice_Flagging/`.

For invoice flagging, the current rule-based target label is:
- `1` if invoice dollars differ from item-level total dollars by more than `2%`
- `1` if `avg_receiving_delay > 10`
- otherwise `0`

The current invoice flagging model is trained on:
- `invoice_quantity`
- `invoice_dollars`
- `Freight`
- `total_item_quantity`
- `total_item_dollars`
- `avg_receiving_delay`

## Project Structure
```text
Invoice_Intelligence/
|-- Freight_Cost_Prediction/   # Regression training and evaluation
|-- Invoice_Flagging/          # Classification training and evaluation
|-- Inference/                 # Inference scripts for trained models
|-- Scripts/                   # Data ingestion and DB utilities
|-- Notebooks/                 # EDA and experimentation
|-- data/                      # SQLite DB and source CSVs
|-- logs/                      # Ingestion logs
|-- models/                    # Saved model and scaler artifacts
|-- README.md
`-- requirements.txt
```

## Inference
Saved artifacts are stored in the root `models/` folder.

Current inference scripts:
- [Inference/predict_freight.py](/C:/WORKS/PROJECTS/Invoice_Intelligence/Inference/predict_freight.py:1)
- [Inference/predict_invoice_flag.py](/C:/WORKS/PROJECTS/Invoice_Intelligence/Inference/predict_invoice_flag.py:1)

The invoice flagging inference script expects these input columns:
- `invoice_quantity`
- `invoice_dollars`
- `Freight`
- `total_item_quantity`
- `total_item_dollars`
- `avg_receiving_delay`

## Future Work
- Add batch prediction support for CSV-based invoice auditing.
- Add a simple UI layer for single-record and batch scoring.
- Capture reviewed outcomes for later retraining and feedback-driven improvement.
