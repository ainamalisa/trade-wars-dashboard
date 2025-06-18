# sections/data.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show():
    st.title("📁 Data Overview")
    st.markdown("---")
    
    st.markdown("""
    ## 📊 Dataset Information
    This section provides an overview of the datasets used in our US-China trade war analysis.
    You can explore each dataset individually to understand the structure and content.
    """)

    # Dataset filenames and user-friendly labels
    dataset_info = {
        "us_china_imports_electronics_clean.csv": "US-China Imports (Electronics)",
        "us_china_exports_electronics_clean.csv": "US-China Exports (Electronics)", 
        "Dataset 3.csv": "Economic Indicators (6 Countries)",
        "combined_sentiment_data.csv": "News Sentiment Analysis",
        "master_modeling_dataset.csv": "Master Modeling Dataset",
        "enhanced_modeling_dataset.csv": "Enhanced Modeling Dataset",
        "time_series_forecasts.csv": "Time Series Forecasts",
        "multicountry_predictions_status_quo.csv": "Status Quo Predictions",
        "multicountry_predictions_tariff_escalation.csv": "Tariff Escalation Predictions",
        "economic_impact_summary.csv": "Economic Impact Summary",
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
        
        st.subheader(f"📁 {selected_label} — Data Preview")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📏 Shape**")
            st.write(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]:,}")

            st.markdown("**🧹 Missing Values (Top 5)**")
            missing_values = df.isnull().sum().sort_values(ascending=False).head(5)
            if missing_values.sum() > 0:
                st.dataframe(missing_values)
            else:
                st.success("No missing values found!")

        with col2:
            st.markdown("**📋 Columns**")
            st.write(f"Total columns: {len(df.columns)}")
            with st.expander("View all column names"):
                for i, col in enumerate(df.columns, 1):
                    st.write(f"{i}. {col}")

        st.markdown("**🔠 Data Types**")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null Count": df.count(),
            "Sample Value": [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A" for col in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Add dataset-specific insights
        st.markdown("### 💡 Dataset Insights")
        
        if "imports" in selected_file.lower():
            st.info("""
            **US-China Imports Dataset:**
            - Contains electronics trade data flowing from China to the US
            - Key for understanding US dependence on Chinese electronics
            - Shows impact of tariffs on import volumes
            """)
            
            if 'value' in df.columns and 'year' in df.columns:
                # Quick visualization
                annual_imports = df.groupby('year')['value'].sum()
                st.markdown("#### 📈 Annual Import Trends")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(annual_imports.index, annual_imports.values, 'b-o', linewidth=2, markersize=6)
                ax.set_title('Annual Electronics Imports from China', fontweight='bold')
                ax.set_xlabel('Year')
                ax.set_ylabel('Import Value (Billions USD)')
                ax.grid(True, alpha=0.3)
                
                # Add trade war indicator
                if annual_imports.index.min() <= 2018 <= annual_imports.index.max():
                    ax.axvline(x=2018, color='red', linestyle='--', alpha=0.7, label='Trade War Start')
                    ax.legend()
                
                st.pyplot(fig)
                
        elif "exports" in selected_file.lower():
            st.info("""
            **US-China Exports Dataset:**
            - Contains electronics trade data flowing from the US to China
            - Shows US export competitiveness in Chinese market
            - Reveals retaliatory effects of trade policies
            """)
            
        elif "dataset 3" in selected_file.lower():
            st.info("""
            **Economic Indicators Dataset:**
            - Multi-country economic data including GDP, inflation, unemployment
            - Essential for understanding broader economic context
            - Used for cross-country comparisons and spillover analysis
            """)
            
            if 'Country Name' in df.columns:
                countries = df['Country Name'].unique()
                st.write(f"**Countries included:** {', '.join(countries)}")
                
        elif "sentiment" in selected_file.lower():
            st.info("""
            **News Sentiment Dataset:**
            - Sentiment analysis of trade war related news articles
            - Captures public opinion and market sentiment
            - Important for understanding policy effectiveness and market reactions
            """)
            
            if 'polarity' in df.columns:
                avg_sentiment = df['polarity'].mean()
                sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                st.metric("Average Sentiment", f"{avg_sentiment:.3f} ({sentiment_label})")
                
        elif "forecast" in selected_file.lower() or "prediction" in selected_file.lower():
            st.info("""
            **Model Output Dataset:**
            - Generated by our predictive models
            - Contains forecasts and scenario predictions
            - Used for policy recommendations and strategic planning
            """)
            
        # Show summary statistics for numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.markdown("### 📊 Summary Statistics (Numerical Columns)")
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
        
    except FileNotFoundError:
        st.error(f"❌ Error: File `{selected_file}` not found in the data directory.")
        st.info("Please ensure the file exists in the 'data/' folder.")
        
    except Exception as e:
        st.error(f"❌ Error loading `{selected_file}`: {e}")
        st.info("This might be due to file format issues or data corruption.")

    # Show overall data summary
    st.markdown("---")
    st.markdown("## 📈 Overall Data Summary")
    
    # Check which files exist
    available_files = []
    missing_files = []
    
    for filename, label in dataset_info.items():
        filepath = os.path.join("data", filename)
        if os.path.exists(filepath):
            available_files.append((filename, label))
        else:
            missing_files.append((filename, label))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Available Datasets")
        if available_files:
            for filename, label in available_files:
                st.write(f"• {label}")
        else:
            st.write("No datasets found")
    
    with col2:
        st.markdown("### ❌ Missing Datasets")
        if missing_files:
            for filename, label in missing_files:
                st.write(f"• {label}")
        else:
            st.write("All datasets are available!")
    
    # Data pipeline overview
    st.markdown("### 🔄 Data Pipeline Overview")
    st.markdown("""
    **Data Flow:**
    1. **Raw Trade Data** → Cleaned and processed trade flows
    2. **Economic Indicators** → Multi-country economic context
    3. **News Sentiment** → Public opinion and market sentiment
    4. **Data Integration** → Master modeling dataset
    5. **Feature Engineering** → Enhanced variables for analysis
    6. **Model Outputs** → Forecasts and predictions
    
    **Key Integration Points:**
    - Trade data aggregated by year for time series analysis
    - Economic indicators matched by country and time period
    - Sentiment scores averaged over relevant time windows
    - All datasets combined for comprehensive modeling
    """)

if __name__ == "__main__":
    show()