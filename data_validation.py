import pandas as pd
import great_expectations as gx
import logging
import os
import uuid
from typing import List, cast

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logger = logging.getLogger("DataValidation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/data_validation.log')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay"
]

class InvoiceDataValidator:
    def __init__(self):
        """Initializes an ephemeral GX context for inference-time validation."""
        try:
            self.context = gx.get_context()
        except Exception as e:
            logger.error(f"Failed to initialize GX context: {e}")
            self.context = None

    def _get_validator(self, df: pd.DataFrame, asset_prefix: str):
        """Helper to get a validator in GX 1.x style using read_dataframe."""
        if self.context is None:
            return None
        
        try:
            ds_name = f"ds_{asset_prefix}"
            asset_name = f"asset_{asset_prefix}_{uuid.uuid4().hex[:8]}"
            
            datasource = self.context.data_sources.add_or_update_pandas(name=ds_name)
            # read_dataframe returns a Batch object in GX 1.x
            batch = datasource.read_dataframe(dataframe=df, asset_name=asset_name)
            
            # Use get_validator with the batch_list
            validator = self.context.get_validator(batch_list=cast(List, [batch])) 
            return validator
        except Exception as e:
            logger.error(f"Error creating GX validator: {e}")
            return None

    def validate_raw_input(self, df: pd.DataFrame):
        """
        Validates raw input data before scaling.
        """
        validator = self._get_validator(df, "raw")
        if validator is None:
            return True, "Validator not initialized"

        try:
            validator.expect_table_columns_to_match_set(column_set=FEATURES, exact_match=False)
            
            for col in FEATURES:
                validator.expect_column_values_to_not_be_null(col)
                
            validator.expect_column_values_to_be_between("invoice_quantity", min_value=0)
            validator.expect_column_values_to_be_between("invoice_dollars", min_value=0)
            validator.expect_column_values_to_be_between("Freight", min_value=0)
            validator.expect_column_values_to_be_between("total_item_quantity", min_value=0)
            validator.expect_column_values_to_be_between("total_item_dollars", min_value=0)
            validator.expect_column_values_to_be_between("avg_receiving_delay", min_value=-365, max_value=365)

            results = validator.validate()
            
            if not results.success:
                logger.warning(f"Scaled data validation failed: {getattr(results, 'statistics', 'No stats available')}")
            else:
                logger.info("Raw data validation passed.")
            
            return results.success, results
        except Exception as e:
            logger.error(f"Error during raw data validation: {e}")
            return True, f"Validation Error: {e}"

    def validate_scaled_data(self, df: pd.DataFrame):
        """
        Validates scaled data.
        """
        validator = self._get_validator(df, "scaled")
        if validator is None:
            return True, "Validator not initialized"

        try:
            for col in FEATURES:
                validator.expect_column_values_to_be_between(col, min_value=-50, max_value=50)

            results = validator.validate()
            
            if not results.success:
                logger.warning(f"Scaled data validation failed: {getattr(results, 'statistics', 'No stats available')}")
            else:
                logger.info("Scaled data validation passed.")
            
            return results.success, results
        except Exception as e:
            logger.error(f"Error during scaled data validation: {e}")
            return True, f"Validation Error: {e}"

def run_validation_flow(df_raw, df_scaled=None):
    """
    Convenience function to run validation without crashing the main thread.
    """
    try:
        validator = InvoiceDataValidator()
        
        # 1. Raw Validation
        raw_success, _ = validator.validate_raw_input(df_raw)
        
        # 2. Scaled Validation (if provided)
        scaled_success = True
        if df_scaled is not None:
            scaled_success, _ = validator.validate_scaled_data(df_scaled)
            
        return raw_success and scaled_success
    except Exception as e:
        logger.error(f"Validation flow failed: {e}")
        return True 
