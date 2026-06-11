from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import logging
import sys
import os

# Ensure root is in path for data_validation import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_validation import run_validation_flow

# Setup local logger
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = Path(__file__).resolve().parent.parent / "models" / "scaler.pkl"
SHAP_BG_PATH = Path(__file__).resolve().parent.parent / "models" / "shap_background.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay"
]

def load_model(model_path = MODEL_PATH):
    with open(model_path,"rb") as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path = SCALER_PATH):
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler

def load_shap_background(path = SHAP_BG_PATH):
    with open(path, "rb") as f:
        bg = joblib.load(f)
    return bg

def predict_invoice_flag(input_data,delay_threshold = 10.0):
    """
    input_data : dict or DataFrame
    delay_threshold: float, user-configurable threshold for receiving delay
    output : pd.DataFrame with predicted flag, risk score, reasons, and SHAP values
    """
    model = load_model()
    scaler = load_scaler()
    
    input_df = pd.DataFrame(input_data)[FEATURES]
    
    # Run Validation
    try:
        # We don't block the pipeline even if validation fails, 
        # but we log it for the auditor/developer to see.
        valid = run_validation_flow(input_df)
        if not valid:
            logger.warning("Input data failed validation checks. Proceeding with caution.")
    except Exception as e:
        logger.error(f"Validation component encountered an error: {e}")

    scaled_input_df_raw = scaler.transform(input_df)
    scaled_input_df = pd.DataFrame(scaled_input_df_raw, columns=FEATURES)
    
    # Optional: Validate scaled data to detect drift/extreme outliers
    try:
        run_validation_flow(input_df, scaled_input_df)
    except:
        pass

    # Get binary prediction
    input_df['Predicted Flag'] = model.predict(scaled_input_df).round()
    
    # Get probability/risk score
    try:
        probabilities = model.predict_proba(scaled_input_df)
        input_df['Risk Score'] = probabilities[:, 1]
    except (AttributeError, IndexError):
        input_df['Risk Score'] = input_df['Predicted Flag'].astype(float)

    # SHAP calculation for the first row (for single prediction UI)
    try:
        import shap
        bg_data = load_shap_background()
        explainer = shap.TreeExplainer(model, bg_data)
        shap_values = explainer.shap_values(scaled_input_df)

        def _to_object_col(arrays):
            """Wrap a list of arrays into a 1-D object-dtype array so pandas
            stores it as a column of arrays rather than trying to broadcast."""
            col = np.empty(len(arrays), dtype=object)
            for i, arr in enumerate(arrays):
                col[i] = arr
            return col

        if isinstance(shap_values, list):
            # Classic RF: shap_values is [neg_class, pos_class]
            input_df['shap_values'] = _to_object_col(
                [shap_values[1][i] for i in range(len(input_df))]
            )
        else:
            if len(shap_values.shape) == 3:  # (samples, features, classes)
                input_df['shap_values'] = _to_object_col(
                    [shap_values[i, :, 1] for i in range(len(input_df))]
                )
            else:                             # (samples, features)
                input_df['shap_values'] = _to_object_col(
                    [shap_values[i] for i in range(len(input_df))]
                )
    except Exception as e:
        print(f"SHAP Error: {e}")
        input_df['shap_values'] = np.empty(len(input_df), dtype=object)

    # Calculating explicit discrepancy reasons
    input_df['dollar_mismatch_pct'] = abs(input_df['invoice_dollars'] -input_df['total_item_dollars']) / input_df['total_item_dollars']

    input_df['delay_flag'] = input_df['avg_receiving_delay'] > delay_threshold
    input_df['delay_flag'] = input_df['delay_flag'].map({True: 'Yes', False: 'No'})

    def generate_reason(row):
        reasons = []
        if row['dollar_mismatch_pct'] > 0.02:
            reasons.append(f"Dollar Mismatch ({(row['dollar_mismatch_pct']*100):.1f}%)")
        if row['delay_flag'] == 'Yes':
            reasons.append(f"High Receiving Delay(> {delay_threshold} days)")
        if not reasons:
            return "No obvious discrepancy"
        return " | ".join(reasons)
    
    input_df['risk_reason'] = input_df.apply(generate_reason, axis=1)
    return input_df

if __name__ == "__main__":
    sample_input = {
        "invoice_quantity": [6, 6],
        "invoice_dollars": [214.26, 219.26],
        "Freight": [3.47, 5.00],
        "total_item_quantity": [6, 6],
        "total_item_dollars": [214.26, 214.26],
        "avg_receiving_delay": [9.00, 12.00]
    }
    prediction = predict_invoice_flag(sample_input)
    print(prediction)
