# FreightAudit

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E.svg)](https://scikit-learn.org/)

FreightAudit is a freight cost forecasting and invoice review project built around a small Streamlit application and two supporting machine learning pipelines. It is designed to help review vendor invoices faster by estimating expected freight charges and flagging invoices that may need manual approval.

## What the project currently does

- Predicts freight cost from invoice dollar value.
- Flags invoices for manual approval using quantity, dollar, freight, and receiving-delay features.
- Supports single-invoice scoring from the Streamlit interface.
- Supports batch CSV scoring and CSV download from the Streamlit interface.
- Skips incomplete batch rows instead of failing the whole upload, and marks skipped rows in the exported output.
- Loads raw CSV files into a SQLite database for training and experimentation.

## Application workflow

The Streamlit app has two main views:

### 1. Freight Cost Prediction

The freight module takes `Invoice Dollars` as input and returns an estimated freight amount using the saved regression model.

### 2. Invoice Manual Approval Flag

The invoice review module supports:

- single invoice prediction through manual form entry
- batch invoice prediction through CSV upload
- downloadable scored CSV output

For batch prediction, the uploaded CSV must contain these columns:

```text
invoice_quantity
invoice_dollars
Freight
total_item_quantity
total_item_dollars
avg_receiving_delay
```

If any of these columns are missing entirely, the app stops and reports the missing columns. If the columns exist but some rows have missing values, the app scores the valid rows and marks the incomplete ones as skipped.

A sample file is included at `sample_invoice.csv`.

## Data source

Training data is loaded into `data/inventory.db` from CSV files under `data/CSVs/`.

The current implementation uses:

- `vendor_invoice` for freight regression
- `vendor_invoice` joined with aggregated `purchases` data for invoice flagging

The SQLite database also contains inventory tables that were loaded during ingestion, although they are not currently used as model features in the app.

## Modeling approach

### Freight prediction

The freight training pipeline compares:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The pipeline evaluates the models on MAE, RMSE, and R-squared, then saves the model with the lowest MAE to `models/predict_freight_model.pkl`.

### Invoice flagging

The invoice flagging pipeline:

- builds training features from invoice and purchase data
- creates a risk label using business rules
- scales numeric inputs with `RobustScaler`
- tunes a `RandomForestClassifier` with `GridSearchCV`
- saves the trained model to `models/predict_flag_invoice.pkl`

The current rule-based target marks an invoice as risky when:

- invoice dollars differ from total item dollars by more than 2%
- average receiving delay is greater than 10 days

## Project structure

```text
.
|-- app.py
|-- sample_invoice.csv
|-- README.md
|-- requirements.txt
|-- data/
|   |-- CSVs/
|   `-- inventory.db
|-- models/
|   |-- predict_freight_model.pkl
|   |-- predict_flag_invoice.pkl
|   `-- scaler.pkl
|-- Freight_Cost_Prediction/
|   |-- data_preprocessing.py
|   |-- model_evaluation.py
|   `-- train.py
|-- Invoice_Flagging/
|   |-- data_preprocessing.py
|   |-- modeling_evaluation.py
|   `-- train.py
|-- Inference/
|   |-- predict_freight.py
|   `-- predict_invoice_flag.py
|-- Scripts/
|   `-- ingestion_db.py
`-- Notebooks/
    |-- Predicting Freight Cost.ipynb
    `-- Invoice Flagging.ipynb
```

## How to run

1. Clone the repository.

```bash
git clone <your-repository-url>
cd Invoice_Intelligence
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Load the raw CSV files into SQLite.

```bash
python Scripts/ingestion_db.py
```

4. Train the models if needed.

```bash
python Freight_Cost_Prediction/train.py
python Invoice_Flagging/train.py
```

Pretrained model files are already present in `models/`.

5. Start the Streamlit app.

```bash
streamlit run app.py
```

## Notes

- `sample_invoice.csv` can be used to test the batch upload flow.
- The current app focuses on prediction and batch scoring. Reviewer feedback capture and model explainability are not implemented yet.

## Author

Souvik Sinha  
GitHub: [DeadMan026](https://github.com/DeadMan026)  
Email: 27souviksinha@gmail.com
