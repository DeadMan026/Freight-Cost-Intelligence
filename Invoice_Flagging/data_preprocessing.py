import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "inventory.db"

def load_invoice_data():
    conn = sqlite3.connect(DB_PATH)
    # query original vendor invoices and purchases
    query = """
    WITH purchase_agg AS (
    SELECT 
    POnumber,
    COUNT(DISTINCT Brand) as total_brands,
    SUM(Quantity) as total_item_quantity,
    SUM(dollars) AS total_item_dollars,
    ROUND(AVG(julianday(ReceivingDate) - julianday(PODate)), 2)  as avg_receiving_delay
    FROM purchases
    GROUP BY POnumber               
    )
                  
    SELECT
    vi.PONumber,
    vi.Quantity AS invoice_quantity,
    vi.Dollars AS invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate)) as days_po_to_invoice,
    (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) as days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay
                   
    FROM vendor_invoice vi 
    LEFT JOIN purchase_agg pa
    on vi.PONumber = pa.PONumber
    """

    df_original = pd.read_sql_query(query,conn)
    
    # query human feedback if table exists
    try:
        df_feedback = pd.read_sql_query("SELECT * FROM auditor_feedback", conn)
    except:
        df_feedback = pd.DataFrame()
    
    conn.close()
    
    # labeling original data
    df_original["flag_invoice"] = df_original.apply(create_invoice_risk_label, axis=1)
    df_original["label_source"] = "RULE"
    
    if not df_feedback.empty:
        # rename for alignment
        df_feedback = df_feedback.rename(columns={"auditor_decision": "flag_invoice"})
        df_feedback["label_source"] = "HUMAN"
        
        # Remove overridden POs from df_original
        overridden_pos = df_feedback['PONumber'].unique()
        df_original = df_original[~df_original['PONumber'].isin(overridden_pos)]
        
        # dropping id/timestamp for merge
        cols_to_keep = [
            "PONumber", "invoice_quantity", "invoice_dollars", "Freight", 
            "total_item_quantity", "total_item_dollars", "avg_receiving_delay", 
            "flag_invoice", "label_source"
        ]
        
        # combine and prioritize human labels via append
        df = pd.concat([df_original[cols_to_keep], df_feedback[cols_to_keep]], ignore_index=True)
        return df
        
    return df_original

def create_invoice_risk_label(row):
   # invoice total mismatch threshold
    mismatch_pct = abs(row["invoice_dollars"] - row["total_item_dollars"]) / row["total_item_dollars"]
    if mismatch_pct > 0.02: 
        return 1
    # high receiving delay threshold
    if row["avg_receiving_delay"] > 10:
        return 1
    return 0    

def split_data(df, features, target):
    x = df[features] 
    y = df[target]
    
    # assign weights based on source
    weights = df["label_source"].map({"RULE": 1, "HUMAN": 5})

    return train_test_split(
        x, y, weights, test_size=0.2, random_state = 42
    )

def scale_features(x_train, x_test, scaler_path):
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    joblib.dump(scaler, scaler_path)
    return x_train_scaled, x_test_scaled
