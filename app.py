import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt

from Inference.predict_freight import predict_freight_cost
from Inference.predict_invoice_flag import predict_invoice_flag

def save_feedback(po_number, prediction, decision, comments=""):
    try:
        conn = sqlite3.connect('data/inventory.db')
        conn.execute(
            "INSERT INTO auditor_feedback (PONumber, prediction, auditor_decision, comments) VALUES (?, ?, ?, ?)",
            (po_number, int(prediction), int(decision), comments)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False

def get_feedback_history():
    try:
        conn = sqlite3.connect('data/inventory.db')
        df = pd.read_sql_query("SELECT * FROM auditor_feedback ORDER BY timestamp DESC LIMIT 10", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

st.set_page_config(
    page_title=' Vendor Invoice Intelligence Portal',
    page_icon='📊',
    layout="wide"
)

st.markdown("""
# Vendor Invoice Intelligence Portal
### Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
""")

st.divider()

st.sidebar.title("Navigation")
selected_model = st.sidebar.radio(
    "Go to",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag",
        "Auditor Review History"
    ]
)

st.sidebar.markdown("""
## Business Impact
- Improved cost forecasting
- Reduced invoice fraud and anomalies
- Faster finance operations
""")

if selected_model == "Freight Cost Prediction":
    st.subheader("Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict freight cost for a vendor invoice using **Invoice Dollars**
    to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight form"):
        dollars = st.number_input(
            "Invoice Dollars",
            min_value=1.0,
            value=20000.0
        )

        submit_freight = st.form_submit_button("Predict Freight Cost")

        if submit_freight:
            input_data = {"Dollars": [dollars]}
            prediction = predict_freight_cost(input_data)["Predicted Freight"]
            st.success("Prediction completed successfully")
            st.metric(label="Estimated Freight Cost", value=f"{prediction[0]:,.2f}")

elif selected_model == "Invoice Manual Approval Flag":
    st.subheader("Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**
    Predict whether a vendor invoice should be **flagged for manual approval**
    based on abnormal cost, freight, or delivery patterns.
    """)

    st.markdown("### Auditor Configuration")
    user_delay_threshold = st.slider(
        "Max Acceptable Receiving Delay (Days)", 
        min_value=1.0, 
        max_value=30.0, 
        value=10.0, 
        step=1.0,
        help="Adjust this threshold to define what constitutes an abnormal receiving delay."
    )
    st.divider()

    st.markdown("### Single Invoice Prediction")

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight = st.number_input("Freight Cost", min_value=0.0, value=1.73)
        with col2:
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=352.95)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=162)
        with col3:
            total_item_dollars = st.number_input("Total Item Dollars", min_value=1.0, value=2476.0)
            avg_receiving_delay = st.number_input("Average Receiving Delay", min_value=0.0, value=9.0)

        submit_flag = st.form_submit_button("Evaluate Invoice Risk")

        if submit_flag:
            input_data = {
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "total_item_quantity": [total_item_quantity],
                "total_item_dollars": [total_item_dollars],
                "avg_receiving_delay": [avg_receiving_delay]
            }

            flag_prediction_df = predict_invoice_flag(input_data, delay_threshold=user_delay_threshold)
            is_flagged = bool(flag_prediction_df['Predicted Flag'].iloc[0])
            risk_score = flag_prediction_df['Risk Score'].iloc[0]
            reason_text = flag_prediction_df['risk_reason'].iloc[0]
            shap_vals = flag_prediction_df['shap_values'].iloc[0]

            st.session_state['last_prediction'] = {
                'prediction': int(is_flagged),
                'risk_score': risk_score,
                'reason': reason_text,
                'shap_values': shap_vals
            }

    if 'last_prediction' in st.session_state:
        pred_data = st.session_state['last_prediction']
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric("ML Risk Score", f"{pred_data['risk_score']*100:.1f}%")
            if pred_data['prediction'] == 1:
                st.error("Status: **Flagged**")
            else:
                st.success("Status: **Safe**")
        
        with col_res2:
            st.info(f"**Discrepancy Reasons:** {pred_data['reason']}")
            
            if pred_data['shap_values'] is not None:
                st.write("**Model Transparency (SHAP Feature Importance)**")
                features = ["Quantity", "Dollars", "Freight", "Total Item Qty", "Total Item $", "Delay"]
                shap_plot_vals = pred_data['shap_values']
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#ff0051' if x > 0 else '#008bfb' for x in shap_plot_vals]
                ax.barh(features, shap_plot_vals, color=colors)
                ax.set_xlabel("Impact on Risk Score")
                ax.set_title("How each feature pushed the risk score up/down")
                st.pyplot(fig)

        st.markdown("#### Auditor Feedback (HITL)")
        col_fb1, col_fb2 = st.columns([1, 1])
        with col_fb1:
            if st.button("✅ Confirm Correct", use_container_width=True):
                if save_feedback("MANUAL_ENTRY", pred_data['prediction'], pred_data['prediction']):
                    st.toast("Feedback saved!", icon="🚀")
        with col_fb2:
            if st.button("❌ Mark as Error", use_container_width=True):
                corrected = 1 - pred_data['prediction']
                if save_feedback("MANUAL_ENTRY", pred_data['prediction'], corrected):
                    st.toast("Correction saved!", icon="🛠️")

    st.divider()
    st.markdown("""
    ### Batch Invoice Prediction
    Upload a CSV of multiple invoices to score them for risk.
    """)

    uploaded_file = st.file_uploader("Upload Invoice CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        required_columns = ["invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars", "avg_receiving_delay"]
        missing_columns = [col for col in required_columns if col not in batch_df.columns]

        if missing_columns:
            st.error(f"Missing required Columns: {missing_columns}")
        else:
            valid_index = batch_df[required_columns].dropna().index
            if not valid_index.empty:
                predictions_df = predict_invoice_flag(batch_df.loc[valid_index], delay_threshold=user_delay_threshold)
                batch_df.loc[valid_index, "Predicted Flag"] = predictions_df["Predicted Flag"].map({1.0: "Manual approval", 0.0: "Safe"})
                batch_df.loc[valid_index, "Risk Score"] = predictions_df["Risk Score"]
                batch_df.loc[valid_index, "risk_reason"] = predictions_df["risk_reason"]

            batch_df["Prediction Status"] = "Skipped (Missing Data)"
            batch_df.loc[valid_index, "Prediction Status"] = "Scored"

            if len(valid_index) > 0:
                st.success(f"Scored {len(valid_index)} rows.")
            st.dataframe(batch_df)
            st.download_button(label="Download Scored CSV", data=batch_df.to_csv(index=False).encode("utf-8"), file_name="scored_invoices.csv", mime="text/csv")

elif selected_model == "Auditor Review History":
    st.subheader("Auditor Review History")
    history_df = get_feedback_history()
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.warning("No feedback history found.")
