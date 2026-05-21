import sqlite3
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "inventory.db"

def load_invoice_data():
    conn = sqlite3.connect(DB_PATH)
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

    df = pd.read_sql_query(query,conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
   # invoice total mismatch -- checks percentage
    mismatch_pct = abs(row["invoice_dollars"] - row["total_item_dollars"]) / row["total_item_dollars"]
    if mismatch_pct > 0.02: # 2% threshold
        return 1
    # Abornormally high receiving delay
    if row["avg_receiving_delay"] > 10:
        return 1
     
    return 0    

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis = 1)
    return df

def split_data(df, features, target):
    x = df[features] 
    y = df[target]

    return train_test_split(
        x, y, test_size=0.2, random_state = 42
    )

def scale_features(x_train, x_test, scaler_path):
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, scaler_path)
    return x_train_scaled, x_test_scaled
