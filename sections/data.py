# sections/data.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

def show():
    st.title("📁 Data Overview")
    st.markdown("---")
    
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1f4e79;">
    <h3 style="color: #1f4e79; margin: 0; font-weight: bold;">📊 How We Collected Our Data</h3>
    <p style="margin: 15px 0 0 0; color: #2c5282; font-size: 16px; line-height: 1.5;">
    Our analysis combines multiple data sources to provide a comprehensive view of the US-China trade war impact. 
    Here's how we gathered and prepared each dataset for analysis.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different data categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trade Volume Data", 
        "💬 Sentiment Analysis", 
        "📊 Economic Indicators", 
        "🔮 Predictive Models"
    ])
    
    with tab1:
        show_trade_data_collection()
    
    with tab2:
        show_sentiment_data_collection()
    
    with tab3:
        show_economic_data_collection()
    
    with tab4:
        show_predictive_data_collection()

def show_trade_data_collection():
    """Explain trade volume data collection process"""
    
    st.subheader("📈 Trade Volume Data Collection")
    
    # Data source explanation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌐 Primary Data Source: WTO Trade Data
        
        **Source:** [WTO Trade and Tariff Data (https://ttd.wto.org/)](https://ttd.wto.org/)
        
        **What we collected:**
        - **US-China Import Data**: Electronics flowing from China to US
        - **US-China Export Data**: Electronics flowing from US to China  
        - **MFN Applied Tariffs**: Most Favored Nation tariff rates
        - **Time Period**: 2018-2024 (focusing on trade war period)
        
        **Why these specific parameters?**
        """)
        
        # Add reasoning boxes
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2d5a2d;">
        <h5 style="color: #1a4d1a; margin: 0;">🗓️ Time Period (2018-2024)</h5>
        <p style="color: #2d5a2d; margin: 5px 0;">
        The US-China trade war officially began in 2018, making this our key analysis period. 
        We included 2024 data to capture the most recent developments and long-term effects.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #856404;">
        <h5 style="color: #856404; margin: 0;">🔌 Electronics Focus</h5>
        <p style="color: #856404; margin: 5px 0;">
        Electronics represent one of the largest trade categories between US and China, 
        with both countries heavily involved in global electronics supply chains. 
        This sector shows the most dramatic tariff impacts.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #d1ecf1; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #0c5460;">
        <h5 style="color: #0c5460; margin: 0;">📊 Data Quality</h5>
        <p style="color: #0c5460; margin: 5px 0;">
        WTO data is official government-reported trade statistics, providing the most 
        reliable and comprehensive view of bilateral trade flows.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### 📋 Key Metrics Collected
        
        **Import Data:**
        - 📦 Trade volumes (USD)
        - 📅 Monthly/annual trends
        - 🏭 Product categories
        - 🌍 Country pairs
        
        **Export Data:**
        - 📤 US shipments to China
        - 💰 Value and quantity
        - 📈 Growth rates
        - 🎯 Market share
        
        **Tariff Data:**
        - 📊 MFN rates
        - 🎯 Applied tariffs
        - 📈 Rate changes over time
        - 🔍 Product-specific rates
        """)
    
    # Show sample data if available
    try:
        imports_df = pd.read_csv('data/Tariff/us_china_imports_electronics_clean.csv')
        exports_df = pd.read_csv('data/Tariff/us_china_exports_electronics_clean.csv')
        
        st.markdown("### 📊 Data Sample Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔽 Imports from China (Sample)**")
            if len(imports_df) > 0:
                sample_imports = imports_df[['year', 'product_code', 'value']].head(3)
                st.dataframe(sample_imports, use_container_width=True)
                
                # Show trade volume trend
                if 'year' in imports_df.columns and 'value' in imports_df.columns:
                    annual_imports = imports_df.groupby('year')['value'].sum() / 1e9
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=annual_imports.index,
                        y=annual_imports.values,
                        mode='lines+markers',
                        name='Imports',
                        line=dict(color='blue', width=3)
                    ))
                    fig.update_layout(
                        title="Annual Electronics Imports from China",
                        xaxis_title="Year",
                        yaxis_title="Value (Billions USD)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**🔼 Exports to China (Sample)**")
            if len(exports_df) > 0:
                sample_exports = exports_df[['year', 'product_code', 'value']].head(3)
                st.dataframe(sample_exports, use_container_width=True)
        
    except FileNotFoundError:
        st.info("💡 Trade data files will be loaded when available in the data directory.")

def show_sentiment_data_collection():
    """Explain sentiment analysis data collection"""
    
    st.subheader("💬 Sentiment Analysis Data Collection")
    
    st.markdown("""
    ### 🗞️ News Data Collection Strategy
    
    We collected news articles from multiple reliable sources to capture public sentiment 
    about the US-China trade war and its impact on electronics trade.
    """)
    
    # Data collection details
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### 🔍 Collection Method
        
        **Primary Sources:**
        - **GNews API**: Real-time news aggregation
        - **RAPIDS**: Financial news and analysis
        - **Time Period**: 2021-2025 (recent sentiment trends)
        
        **Search Keywords Used:**
        - 🏛️ **"Trump"** - Policy maker influence
        - 📊 **"Tariff"** - Direct policy measures  
        - 🤝 **"Trade"** - General trade relations
        - 🔌 **"Electronic"** - Sector-specific impact
        
        **Quality Control:**
        - ✅ Only reliable news sources included
        - ✅ Duplicate articles removed
        - ✅ Language filtering (English only)
        - ✅ Relevance scoring applied
        """)
        
        # Why these keywords
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #6c757d;">
        <h5 style="color: #495057; margin: 0;">🎯 Keyword Strategy</h5>
        <p style="color: #6c757d; margin: 5px 0;">
        <strong>Trump:</strong> Captures policy-related sentiment and political discussions<br>
        <strong>Tariff:</strong> Direct mentions of trade policy tools<br>
        <strong>Trade:</strong> Broad coverage of US-China commercial relations<br>
        <strong>Electronic:</strong> Sector-specific impact and industry concerns
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### 📊 Sentiment Processing
        
        **Analysis Steps:**
        1. **Text Cleaning**
           - Remove noise and formatting
           - Standardize language
        
        2. **Sentiment Scoring**
           - Polarity: -1 (negative) to +1 (positive)
           - Subjectivity: 0 (objective) to 1 (subjective)
        
        3. **Classification**
           - Positive sentiment
           - Negative sentiment  
           - Neutral sentiment
        
        4. **Aggregation**
           - Daily averages
           - Monthly trends
           - Event-based analysis
        """)
    
    # Show sample sentiment data if available
    try:
        sentiment_df = pd.read_csv('data/combined_sentiment_data.csv')
        
        st.markdown("### 📈 Sentiment Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_sentiment = sentiment_df['polarity'].mean()
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}", help=f"Overall sentiment is {sentiment_label}")
        
        with col2:
            total_articles = len(sentiment_df)
            st.metric("Total Articles", f"{total_articles:,}", help="Number of news articles analyzed")
        
        with col3:
            date_range = f"{sentiment_df['published_date'].min()} to {sentiment_df['published_date'].max()}"
            st.metric("Date Range", "2021-2025", help="Period covered by sentiment analysis")
        
        # Sentiment distribution
        if len(sentiment_df) > 0:
            sentiment_counts = sentiment_df['sentiment'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution in News Articles",
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545', 
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.info("💡 Sentiment data will be displayed when files are available.")

def show_economic_data_collection():
    """Explain economic indicators data collection"""
    
    st.subheader("📊 Economic Indicators Data Collection")
    
    st.markdown("""
    ### 🌍 Multi-Country Economic Analysis
    
    To understand the broader impact of US-China trade tensions, we collected economic 
    indicators from multiple countries involved in semiconductor and electronics trade.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📈 Data Source: World Development Indicators
        
        **Primary Source:** World Bank World Development Indicators
        - Most comprehensive global economic database
        - Standardized metrics across all countries
        - Regular updates and historical data
        
        **Key Economic Indicators Collected:**
        - 📊 **GDP Growth Rate** - Economic expansion/contraction
        - 💰 **Inflation Rate** - Price level changes
        - 👥 **Unemployment Rate** - Labor market conditions
        - 📈 **Stock Market Performance** - Investor sentiment
        - 🏛️ **Tariff Rates** - Trade policy measures
        """)
        
        # Country selection rationale
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #1976d2;">
        <h5 style="color: #1565c0; margin: 0;">🌏 Why These Countries?</h5>
        <div style="color: #1976d2; margin: 10px 0;">
        <p><strong>🇺🇸 United States & 🇨🇳 China:</strong> Primary trade war participants</p>
        <p><strong>🇲🇾 Malaysia:</strong> Major semiconductor assembly hub in Southeast Asia</p>
        <p><strong>🇩🇪 Germany:</strong> European manufacturing powerhouse and electronics exporter</p>
        <p><strong>🇰🇷 South Korea:</strong> Global leader in memory chips and electronics</p>
        <p><strong>🇻🇳 Vietnam:</strong> Emerging manufacturing alternative to China</p>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### 🎯 Analysis Focus
        
        **Semiconductor Trade Impact:**
        - Supply chain disruptions
        - Manufacturing shifts
        - Investment flows
        
        **Economic Spillovers:**
        - GDP growth effects
        - Employment impacts
        - Inflation pressures
        
        **Policy Responses:**
        - Tariff adjustments
        - Trade agreements
        - Industrial policies
        """)
    
    # Show economic data sample if available
    try:
        economic_df = pd.read_csv('data/Dataset 3.csv')
        
        st.markdown("### 📊 Economic Data Overview")
        
        if 'Country Name' in economic_df.columns:
            countries = economic_df['Country Name'].unique()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌍 Countries in Dataset:**")
                for country in countries:
                    st.write(f"• {country}")
            
            with col2:
                if 'Time' in economic_df.columns:
                    time_range = f"{economic_df['Time'].min()} - {economic_df['Time'].max()}"
                    st.metric("Time Period", time_range)
                
                total_indicators = len([col for col in economic_df.columns if '[' in col])
                st.metric("Economic Indicators", total_indicators)
    
    except FileNotFoundError:
        st.info("💡 Economic indicators data will be displayed when available.")

def show_predictive_data_collection():
    """Explain predictive modeling data preparation"""
    
    st.subheader("🔮 Predictive Modeling Data Integration")
    
    st.markdown("""
    ### 🔄 Data Integration Process
    
    Our predictive models combine all the data sources above to create comprehensive 
    forecasts of future trade impacts.
    """)
    
    # Data integration flow
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h4 style="color: #495057; margin: 0; text-align: center;">📊 Data Integration Pipeline</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Create flow diagram
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; text-align: center;">
        <h5 style="color: #1565c0; margin: 0;">📈 Trade Data</h5>
        <p style="color: #1976d2; margin: 5px 0; font-size: 14px;">
        • Import/Export volumes<br>
        • Tariff rates<br>
        • Product categories
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f3e5f5; padding: 15px; border-radius: 10px; text-align: center;">
        <h5 style="color: #7b1fa2; margin: 0;">💬 Sentiment</h5>
        <p style="color: #8e24aa; margin: 5px 0; font-size: 14px;">
        • News polarity<br>
        • Public opinion<br>
        • Market sentiment
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; text-align: center;">
        <h5 style="color: #2e7d32; margin: 0;">📊 Economics</h5>
        <p style="color: #388e3c; margin: 5px 0; font-size: 14px;">
        • GDP growth<br>
        • Unemployment<br>
        • Inflation rates
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; text-align: center;">
        <h5 style="color: #f57c00; margin: 0;">🔮 Predictions</h5>
        <p style="color: #fb8c00; margin: 5px 0; font-size: 14px;">
        • Future forecasts<br>
        • Scenario analysis<br>
        • Policy impacts
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Integration details
    st.markdown("""
    ### 🛠️ Data Preparation Steps
    
    **1. Data Cleaning & Standardization**
    - Remove missing values and outliers
    - Standardize date formats and country names
    - Convert currencies to consistent units (USD)
    
    **2. Feature Engineering**
    - Create lagged variables for time series analysis
    - Calculate growth rates and percentage changes
    - Generate interaction terms between variables
    
    **3. Data Integration**
    - Merge datasets by country and time period
    - Create master modeling dataset
    - Validate data consistency across sources
    
    **4. Model-Ready Datasets**
    - Time series forecasting data
    - Cross-country panel data
    - Sentiment-enhanced economic indicators
    """)
    
    # Show model datasets if available
    try:
        master_df = pd.read_csv('data/Result/master_modeling_dataset.csv')
        
        st.markdown("### 📊 Final Modeling Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Variables", len(master_df.columns), help="Combined features from all data sources")
        
        with col2:
            st.metric("Time Periods", len(master_df), help="Years of data available for modeling")
        
        with col3:
            completeness = (master_df.notna().sum().sum() / (len(master_df) * len(master_df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%", help="Percentage of non-missing values")
        
        # Show data quality over time
        if 'year' in master_df.columns:
            st.markdown("**📈 Data Availability by Year:**")
            
            yearly_completeness = []
            for year in master_df['year']:
                year_row = master_df[master_df['year'] == year]
                completeness = (year_row.notna().sum().sum() / (len(year_row) * len(year_row.columns))) * 100
                yearly_completeness.append({'Year': year, 'Completeness': completeness})
            
            completeness_df = pd.DataFrame(yearly_completeness)
            
            fig = px.bar(
                completeness_df,
                x='Year',
                y='Completeness',
                title="Data Completeness by Year",
                color='Completeness',
                color_continuous_scale='RdYlGn',
                range_color=[80, 100]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    except FileNotFoundError:
        st.info("💡 Model datasets will be displayed when available.")
    
    # Data summary
    st.markdown("---")
    st.markdown("""
    ### 🎯 Data Collection Summary
    
    **✅ What We Achieved:**
    - **Comprehensive Coverage**: 7 years of trade war data (2018-2024)
    - **Multi-Source Integration**: Trade, economic, and sentiment data combined
    - **Global Perspective**: 6 countries representing key trade relationships
    - **Real-Time Insights**: Recent sentiment analysis through 2025
    
    **🔄 Continuous Updates:**
    - Monthly trade statistics updates
    - Daily sentiment monitoring
    - Quarterly economic indicator releases
    - Real-time policy change tracking
    
    This comprehensive data foundation enables robust analysis and reliable predictions 
    about the ongoing impacts of US-China trade policies.
    """)

if __name__ == "__main__":
    show()