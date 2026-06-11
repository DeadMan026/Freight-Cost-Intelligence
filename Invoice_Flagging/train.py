import shutil
from pathlib import Path
from data_preprocessing import load_invoice_data, split_data, scale_features
from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib
import sys
import os
import logging

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_validation import run_validation_flow

# Setup logging
logger = logging.getLogger(__name__)

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
    # model storage paths
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'predict_flag_invoice.pkl'
    scaler_path = model_dir / 'scaler.pkl'
    shap_bg_path = model_dir / 'shap_background.pkl'

    # model backup / versioning
    if model_path.exists():
        backup_path = model_dir / 'predict_flag_invoice_backup.pkl'
        shutil.copy(model_path, backup_path)
        # backup existing model

    # data ingestion and splitting with weights
    df = load_invoice_data()
    
    # Run Validation before training
    try:
        logger.info("Running data validation on training set...")
        valid = run_validation_flow(df[FEATURES])
        if not valid:
            logger.warning("Training data failed validation checks. Results may be unreliable.")
        else:
            logger.info("Training data passed validation.")
    except Exception as e:
        logger.error(f"Validation failed during training: {e}")

    x_train, x_test, y_train, y_test, w_train, w_test = split_data(df, FEATURES, TARGET)
    
    # scaling features
    x_train_scaled, x_test_scaled = scale_features(
        x_train, x_test, str(scaler_path)
    ) 

    # model training with human-label priority weights
    grid_search = train_random_forest(x_train_scaled, y_train)
    # retraining with weights for final estimator
    best_model = grid_search.best_estimator_
    best_model.fit(x_train_scaled, y_train, sample_weight=w_train)
    
    # evaluation logic
    evaluate_classifier(
        best_model,
        x_test_scaled,
        y_test,
        'Random Forest Classifier'
    )

    # saving artifacts
    joblib.dump(best_model, model_path)
    joblib.dump(x_train_scaled[:100], shap_bg_path)
    # saving shap background

if __name__ == "__main__":
    main()
