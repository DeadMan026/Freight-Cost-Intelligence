from pathlib import Path
import joblib
import pandas as pd

# Path to the model in the root 'models' directory
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "predict_freight_model.pkl"

def load_model(model_path=MODEL_PATH):
    """load the trained freight cost prediction model"""
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    predicts freight cost for new vendor invoices
    input data : dict
    output : dictpd.DataFrame with preficting freight cost
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    # Local Inference run
    sample_data = {
        "Dollars" : [3000, 5000, 8000]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)

    