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

    #load data
    df = load_invoice_data()
    df = apply_labels(df)

    # apply labels
    x_train, x_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    
    scaler_path = model_dir / 'scaler.pkl'
    x_train_scaled, x_test_scaled = scale_features(
        x_train, x_test, str(scaler_path)
    ) 

    # train and evaluate models
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

if __name__ == "__main__":
    main()
