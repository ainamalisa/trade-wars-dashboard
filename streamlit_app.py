import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.linear_model import LinearRegression

# Streamlit page config
st.set_page_config(page_title="TradeWars Dashboard", layout="wide")

# Sidebar navigation
# --- CSS to make buttons full-width and borderless ---
st.sidebar.markdown("""
    <style>
    /* Make buttons full width and remove border */
    div.stButton > button {
        width: 100% !important;
        border: none !important;
        border-radius: 8px !important;
        background-color: transparent !important;
        padding-left: 1rem !important;  /* Padding for left alignment */
        text-align: left !important;     /* Left align text */
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        transition: background-color 0.3s ease !important;
    }
    /* Change background on hover */
    div.stButton > button:hover {
        background-color: #d6e4f0 !important;
        color: black !important;
    }
    

    </style>
""", unsafe_allow_html=True)

# --- Section configuration ---
sections = {
    "📌 Project Overview": "Project Overview",
    "📁 Data Overview": "Data Overview",
    "📈 Trends in Trade Volume": "Trends in Trade Volume",
    "💬 Sentiment Analysis": "Sentiment Analysis",
    "📊 Correlation Analysis": "Correlation Analysis",
    "📉 Predictive Modeling": "Predictive Modeling",
    "✅ Conclusion & Recommendations": "Conclusion & Recommendations"
}

# --- Initialize session state ---
if "active_section" not in st.session_state:
    st.session_state.active_section = "Project Overview"

# --- Sidebar layout ---
st.sidebar.title("📊 TradeWars Dashboard")

for label, key in sections.items():
    if st.sidebar.button(label):
        st.session_state.active_section = key

# --- Use this variable to control main page content ---
section = st.session_state.active_section

# --- SECTION 1: Project Overview ---
if section == "Project Overview":
    st.title("TradeWars: Impact Analysis of US-China Tariffs")
    st.markdown("""
    Welcome to the TradeWars dashboard.  
    This project analyzes the **impact of recent US-China tariffs** on global trade and the economy, through data-driven techniques including:
    // KENA TUKAR
    - Trade trend analysis  
    - Sentiment analysis from news and reports  
    - Correlation of tariffs with key economic indicators  
    - Predictive modeling  
    - Interactive visualizations & recommendations
    """)

# --- SECTION 2: Data Overview ---
if section == "Data Overview":
    st.header("Data Overview")
    st.markdown("LETAK EDA COMBINED DATA??")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("File uploaded successfully!")
        st.write("First 5 rows of data:")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Missing values:")
        st.write(df.isnull().sum())

# --- SECTION 3: Trends in Trade Volume ---
if section == "Trends in Trade Volume":
    st.header("Trends in Trade Volume")

    try:
        # Load data
        df = pd.read_csv("Tariff electrical component.csv")

        # Show raw data table
        # st.subheader("📄 Raw Tariff Data")
        # st.plotly_chart(go.Figure(data=[go.Table(
        #     header=dict(values=list(df.columns), fill_color='lightblue', align='left'),
        #     cells=dict(values=[df[col] for col in df.columns], fill_color='white', align='left'))
        # ]))

        # Data cleaning
        df = df.dropna(subset=['year', 'value_usd', 'mfn_avg_duty'])
        df = df[df['importer'] != 'ALL Available Markets']
        df['year'] = df['year'].astype(int)

        df['value_usd_cleaned'] = df['value_usd'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df['value_usd_cleaned'] = pd.to_numeric(df['value_usd_cleaned'], errors='coerce')
        df = df.dropna(subset=['value_usd_cleaned'])
        df['value_billion_usd'] = df['value_usd_cleaned'] / 1_000_000_000

        numeric_cols = ['mfn_avg_duty', 'best_avg_duty', 'mfn_line_count']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Show cleaned data
        # st.subheader("🧹 Cleaned Tariff Data")
        # st.plotly_chart(go.Figure(data=[go.Table(
        #     header=dict(values=list(df.columns), fill_color='lightblue', align='left'),
        #     cells=dict(values=[df[col] for col in df.columns], fill_color='white', align='left'))
        # ]))

        # 1. Trade Volume Over Time
        st.subheader("Trade Volume Over Time")
        trade_by_year = df.groupby('year')['value_billion_usd'].sum().reset_index()
        fig1 = px.line(trade_by_year, x='year', y='value_billion_usd',
                       title='Trade Volume Over Time',
                       labels={'value_billion_usd': 'Trade Value (Billion USD)', 'year': 'Year'})
        st.plotly_chart(fig1)

        # 2. Average Tariff Rate Over Time
        st.subheader("Average MFN Tariff Rate Over Time")
        avg_tariff_by_year = df.groupby('year')['mfn_avg_duty'].mean().reset_index()
        fig2 = px.line(avg_tariff_by_year, x='year', y='mfn_avg_duty',
                       title='Average MFN Tariff Rate Over Time',
                       labels={'mfn_avg_duty': 'Avg MFN Tariff (%)', 'year': 'Year'})
        st.plotly_chart(fig2)

        # 3. Bar Plot: Yearly Trade Volume
        st.subheader("Yearly Trade Volume (Bar Chart)")
        fig3 = px.bar(trade_by_year, x='year', y='value_billion_usd',
                      title='Yearly Trade Volume',
                      labels={'value_billion_usd': 'Trade Value (Billion USD)', 'year': 'Year'})
        st.plotly_chart(fig3)

        # 4. Top 20 Importers by Trade Value
        st.subheader("Top 20 Importers by Trade Value")
        trade_by_importer = df.groupby(['exporter', 'importer'])['value_billion_usd'].sum().reset_index()
        top_importers = trade_by_importer.sort_values(by='value_billion_usd', ascending=False).head(20)

        fig4 = px.bar(top_importers, x='importer', y='value_billion_usd', color='exporter',
                      title='Top 20 Importers by Trade Value',
                      labels={'value_billion_usd': 'Trade Value (Billion USD)', 'importer': 'Importer'},
                      barmode='group')
        st.plotly_chart(fig4)

    except FileNotFoundError:
        st.error("Tariff electrical component CSV file not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")

# --- SECTION 4: Sentiment Analysis ---
if section == "Sentiment Analysis":
    st.header("Sentiment Analysis on News Articles")
    st.markdown("LETAK SENTIMENT ANALYSIS")
    st.markdown("Paste or upload a news article/report below to analyze its sentiment regarding US-China tariffs.")
    text = st.text_area("Enter news excerpt or paragraph:")

    if text:
        analysis = TextBlob(text)
        polarity = analysis.polarity
        subjectivity = analysis.subjectivity

        st.metric(label="Polarity", value=f"{polarity:.2f}")
        st.metric(label="Subjectivity", value=f"{subjectivity:.2f}")

        if polarity > 0:
            st.success("Sentiment: Positive")
        elif polarity < 0:
            st.error("Sentiment: Negative")
        else:
            st.info("Sentiment: Neutral")

# --- SECTION 5: Correlation Analysis ---
if section == "Correlation Analysis":
    st.header("Correlation Analysis Between Tariff Rates and Economic Indicators")

    try:
        data = pd.read_csv("Dataset 3.csv")

        # Clean data
        data.replace('..', np.nan, inplace=True)
        rows_to_drop = data[data['Time'].between(2015, 2017)].index
        data = data.drop(rows_to_drop)

        for col in data.columns[4:]:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        countries = data['Country Name'].unique()
        country_dfs = {}

        for country in countries:
            country_df = data[data['Country Name'] == country].copy()
            for col in country_df.columns[4:]:
                if country_df[col].notna().sum() > 0:
                    median_value = country_df[col].median()
                else:
                    median_value = 0
                country_df[col] = country_df[col].fillna(median_value)
            country_dfs[country] = country_df

        merged_df = pd.concat(country_dfs.values(), ignore_index=True)

        # Define indicators and target column
        indicators = {
            'GDP Growth': 'GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]',
            'Inflation Rate': 'Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]',
            'Stock Market': 'S&P Global Equity Indices (annual % change) [CM.MKT.INDX.ZG]',
            'Employment Rate': 'Unemployment, total (% of total labor force) (modeled ILO estimate) [SL.UEM.TOTL.ZS]'
        }
        target_col = 'Tariff rate, applied, weighted mean, all products (%) [TM.TAX.MRCH.WM.AR.ZS]'

        # Ensure numeric conversion
        for col in list(indicators.values()) + [target_col]:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

        # Calculate correlation per country
        indicator_correlations = {indicator: [] for indicator in indicators}

        for country in merged_df['Country Name'].unique():
            country_data = merged_df[merged_df['Country Name'] == country]
            for indicator, col_name in indicators.items():
                x = country_data[col_name]
                y = country_data[target_col]
                valid = x.notna() & y.notna()
                correlation = x[valid].corr(y[valid]) if valid.sum() > 1 else np.nan
                indicator_correlations[indicator].append((country, correlation))

        # Display 4 indicator correlation barplots
        st.subheader("Correlation Between Tariff Rate and Economic Indicators by Country")

        for indicator, correlations in indicator_correlations.items():
            corr_df = pd.DataFrame(correlations, columns=['Country', 'Correlation']).dropna().sort_values('Correlation', ascending=False)
            
            fig = px.bar(
                corr_df,
                x='Correlation',
                y='Country',
                orientation='h',
                color='Correlation',
                color_continuous_scale='Viridis',
                title=f'Correlation between Tariff and {indicator}',
                range_x=[-1, 1]
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)


        # Heatmap of all correlations
        st.subheader("Heatmap: Correlation Between Tariff Rates and Economic Indicators Across Countries")

        # Prepare the heatmap DataFrame
        heatmap_data = {indicator: dict(vals) for indicator, vals in indicator_correlations.items()}
        heatmap_df = pd.DataFrame(heatmap_data)

        # Round values for annotation
        z_text = heatmap_df.round(2).values.tolist()

        # Create heatmap with annotations
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns.tolist(),
            y=heatmap_df.index.tolist(),
            text=z_text,
            texttemplate="%{text}",
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation', tickvals=[-1, -0.5, 0, 0.5, 1]),
            hovertemplate='Country: %{y}<br>Indicator: %{x}<br>Correlation: %{z:.2f}<extra></extra>',
        ))

        fig.update_layout(
            xaxis_title='Economic Indicator',
            yaxis_title='Country',
            xaxis=dict(tickangle=-45, automargin=True),
            yaxis=dict(automargin=True),
            height=900,
            margin=dict(l=120, r=40, t=80, b=120),
        )

        st.plotly_chart(fig, use_container_width=True)



    except Exception as e:
        st.error(f"Error during correlation analysis: {e}")


# --- SECTION 7: Conclusion & Recommendations ---
if section == "Conclusion & Recommendations":
    st.header("Conclusion & Recommendations")
    st.markdown("KENA TUKAR NANTI")
    st.markdown("""
    ### Summary of Findings
    - 📉 **Trade volumes** decreased significantly after tariffs were enforced.
    - 🧑‍🌾 **Agriculture** and 💻 **Technology** sectors were among the most affected.
    - 📰 Sentiment from news and reports skewed mostly **negative**.
    - 📉 **Correlation** found between tariff changes and key indicators like GDP, inflation, and employment.

    ### Recommendations
    - 🌐 **Diversify global trade** partners to reduce dependency.
    - 📊 Use **predictive analytics** for future policy evaluation.
    - 🏛️ Policy makers should implement **data-driven** trade policies.
    - 💼 Businesses should develop **resilient supply chains** and hedge against geopolitical risks.
    """)
