from pathlib import Path
from data_preprocessing import load_invoice_data, create_invoice_risk_label, apply_labels, split_data, scale_features
from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay"
]

TARGET = 'flag_invoice'

def main():
    # Path to root models directory
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    df = load_invoice_data()
    df = apply_labels(df)


    x_train, x_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    
    scaler_path = model_dir / 'scaler.pkl'
    x_train_scaled, x_test_scaled = scale_features(
        x_train, x_test, str(scaler_path)
    ) 

    grid_search = train_random_forest(x_train_scaled, y_train)
    evaluate_classifier(
        grid_search.best_estimator_,
        x_test_scaled,
        y_test,
        'Random Forest Classifier'
    )

    # save best model
    model_path = model_dir / 'predict_flag_invoice.pkl'
    joblib.dump(grid_search.best_estimator_, model_path)
    
    # Save a sample of training data for SHAP background
  
    shap_bg_path = model_dir / 'shap_background.pkl'
    joblib.dump(x_train_scaled[:100], shap_bg_path)
    print(f"SHAP background data saved to {shap_bg_path}")

if __name__ == "__main__":
    main()
