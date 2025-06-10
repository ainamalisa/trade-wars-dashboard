import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.header("Trends in Trade Volume")
    
    @st.cache_data
    def load_data():
        return pd.read_csv("data/Tariff electrical component.csv")
    
    try:
        df = load_data()
        
        # Data cleaning
        df = df.dropna(subset=['year', 'value_usd', 'mfn_avg_duty'])
        df = df[df['importer'] != 'ALL Available Markets']
        df['year'] = df['year'].astype(int)
        df['value_usd_cleaned'] = df['value_usd'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df['value_usd_cleaned'] = pd.to_numeric(df['value_usd_cleaned'], errors='coerce')
        df = df.dropna(subset=['value_usd_cleaned'])
        df['value_billion_usd'] = df['value_usd_cleaned'] / 1_000_000_000
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["Trade Volume", "Tariff Rates", "Top Importers"])
        
        with tab1:
            st.subheader("Trade Volume Over Time")
            trade_by_year = df.groupby('year')['value_billion_usd'].sum().reset_index()
            fig1 = px.line(trade_by_year, x='year', y='value_billion_usd',
                          title='Trade Volume Over Time',
                          labels={'value_billion_usd': 'Trade Value (Billion USD)', 'year': 'Year'})
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab2:
            st.subheader("Tariff Rates Over Time")
            avg_tariff_by_year = df.groupby('year')['mfn_avg_duty'].mean().reset_index()
            fig2 = px.line(avg_tariff_by_year, x='year', y='mfn_avg_duty',
                          title='Average MFN Tariff Rate Over Time',
                          labels={'mfn_avg_duty': 'Avg MFN Tariff (%)', 'year': 'Year'})
            st.plotly_chart(fig2, use_container_width=True)
            
        with tab3:
            st.subheader("Top Importers")
            trade_by_importer = df.groupby(['exporter', 'importer'])['value_billion_usd'].sum().reset_index()
            top_importers = trade_by_importer.sort_values(by='value_billion_usd', ascending=False).head(20)
            
            fig3 = px.bar(top_importers, x='importer', y='value_billion_usd', color='exporter',
                         title='Top 20 Importers by Trade Value',
                         labels={'value_billion_usd': 'Trade Value (Billion USD)', 'importer': 'Importer'})
            st.plotly_chart(fig3, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")