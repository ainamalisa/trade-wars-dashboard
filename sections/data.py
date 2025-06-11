import streamlit as st
import pandas as pd
import os

def show():
    st.header("📊 Data Overview")

    # Dataset filenames and user-friendly labels
    dataset_info = {
        "us_china_imports_electronics_clean.csv": "US-China Imports (Electronics)",
        "us_china_exports_electronics_clean.csv": "US-China Exports (Electronics)",
        "Dataset 3.csv": "Dataset 3 (Unnamed)",
        "master_modeling_dataset.csv": "Master Modeling Dataset",
        "enhanced_modeling_dataset.csv": "Enhanced Modeling Dataset",
        "time_series_forecasts.csv": "Time Series Forecasts",
        "multicountry_predictions_status_quo.csv": "Status Quo Predictions",
        "multicountry_predictions_tariff_escalation.csv": "Tariff Escalation Predictions",
        "economic_impact_summary.csv": "Economic Impact Summary",
        "combined_sentiment_data.csv": "Combined Sentiment Data",
        "Tariff electrical component.csv": "Tariff - Electrical Components"
    }

    # Reverse lookup: label → filename
    label_to_file = {v: k for k, v in dataset_info.items()}

    # Dropdown selection
    selected_label = st.selectbox("Select a dataset to view", list(dataset_info.values()))
    selected_file = label_to_file[selected_label]
    filepath = os.path.join("data", selected_file)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        st.error(f"❌ Error loading `{selected_file}`: {e}")
        return

    st.subheader(f"📁 {selected_label} — Data Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📏 Shape**")
        st.write(df.shape)

        st.markdown("**🧹 Missing Values (Top 5)**")
        st.dataframe(df.isnull().sum().sort_values(ascending=False).head(5))

    with col2:
        st.markdown("**📋 Columns**")
        st.write("Columns:", df.columns.tolist())

    st.markdown("**🔠 Data Types**")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.values
    })
    st.table(dtype_df)
