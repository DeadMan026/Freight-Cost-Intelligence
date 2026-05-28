import streamlit as st
import numpy as np
import pandas as pd

from Inference.predict_freight import predict_freight_cost
from Inference.predict_invoice_flag import predict_invoice_flag

# --------------------------------------
#  PAGE CONFIGURATION
# --------------------------------------

st.set_page_config(
    page_title=' Vendor Invoice Intelligence Portal',
    page_icon='📊',
    layout="wide"
)

# ---------------------------------------
#  HEADER
# ---------------------------------------

st.markdown("""
# Vendor Invoice Intelligence Portal
### Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
""")

st.divider()

# ----------------------------------------
# SIDEBAR
# ----------------------------------------
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose prediction module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
## Business Impact

- Improved cost forecasting
- Reduced invoice fraud and anomalies
- Faster finance operations
""")

# ----------------------------------------
# FREIGHT COST PREDICTION
# ----------------------------------------
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
            input_data = {
                "Dollars": [dollars]
            }

            prediction = predict_freight_cost(input_data)["Predicted Freight"]
            st.success("Prediction completed successfully")

            st.metric(
                label="Estimated Freight Cost",
                value=f"{prediction[0]:,.2f}"
            )

# --------------------------------------------------------
# Invoice Flag Prediction
# --------------------------------------------------------

else:
    st.subheader("Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**
    Predict whether a vendor invoice should be **flagged for manual approval**
    based on abnormal cost, freight, or delivery patterns.
    """)

    st.markdown("""
    ### Single Invoice Prediction
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )

        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

            avg_receiving_delay = st.number_input(
                "Average Receiving Delay",
                min_value=0.0,
                value=9.0
            )

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

            flag_prediction = predict_invoice_flag(input_data)['Predicted Flag']

            is_flagged = bool(flag_prediction[0])

            if is_flagged:
                st.error("Invoice requires **manual approval**")
            else:
                st.success("Invoice is **safe for auto-approval**")

    st.divider()
    st.markdown("""
    ### Batch Invoice Prediction

    Upload a CSV of multiple invoices and score them all at once for risk.
    Then download the scored CSV with predicted flags for quick review and action.
    """)

    uploaded_file = st.file_uploader(
        "Upload Invoice CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        required_columns = [
            "invoice_quantity",
            "invoice_dollars",
            "Freight",
            "total_item_quantity",
            "total_item_dollars",
            "avg_receiving_delay"
        ]

        missing_columns = [col for col in required_columns if col not in batch_df.columns]

        if missing_columns:
            st.error(f"Missing required Columns: {missing_columns}")
        else:
            # identifying rows that have all required values
            valid_index = batch_df[required_columns].dropna().index

            if not valid_index.empty:
                predictions_df = predict_invoice_flag(batch_df.loc[valid_index])

                # mapping predictions back to original df
                batch_df.loc[valid_index, "Predicted Flag"] = predictions_df["Predicted Flag"].map({
                    1.0: "Manual approval",
                    0.0: "Safe"
                })

            # adding status labels and count results
            batch_df["Prediction Status"] = "Skipped (Missing Data)"
            batch_df.loc[valid_index, "Prediction Status"] = "Scored"

            scored_count = len(valid_index)
            skipped_count = len(batch_df) - scored_count

            if scored_count > 0:
                st.success(f"Scored {scored_count} rows successfully.")
            if skipped_count > 0:
                st.warning(f"Skipped {skipped_count} rows due to missing fields.")

            st.dataframe(batch_df)

            csv_output = batch_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Scored CSV",
                data=csv_output,
                file_name="FreightAudit_scored_invoices.csv",
                mime="text/csv"
            )
