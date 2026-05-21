from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = Path(__file__).resolve().parent.parent / "models" / "scaler.pkl"
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay"
]

"""
we cant pass raw data to the model as the model was trained on scaled data, so we also import the scaler.
also we must ensure during inference that we feed all 6 columns in exact order as it was during training(model -> dont know column name, just col 0, col 1 ..)
"""

def load_model(model_path = MODEL_PATH):
    with open(model_path,"rb") as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path = SCALER_PATH):
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler

def predict_invoice_flag(input_data):
    """
    input_data : dict
    output : pd.DataFrame with predicted flag
    """
    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)[FEATURES]
    scaled_input_df = scaler.transform(input_df)
    input_df['Predicted Flag'] = model.predict(scaled_input_df).round()
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
