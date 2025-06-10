import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.header("Correlation Analysis Between Tariff Rates and Economic Indicators")
    
    @st.cache_data
    def load_data():
        data = pd.read_csv("data/Dataset 3.csv")
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

        data = pd.concat(country_dfs.values(), ignore_index=True)
            
        return data
    
    try:
        data = load_data()
        
        # Define indicators
        indicators = {
            'GDP Growth': 'GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]',
            'Inflation Rate': 'Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]',
            'Stock Market': 'S&P Global Equity Indices (annual % change) [CM.MKT.INDX.ZG]',
            'Employment Rate': 'Unemployment, total (% of total labor force) (modeled ILO estimate) [SL.UEM.TOTL.ZS]'
        }
        target_col = 'Tariff rate, applied, weighted mean, all products (%) [TM.TAX.MRCH.WM.AR.ZS]'
        
        #Ensure numeric conversion
        for col in list(indicators.values()) + [target_col]:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Calculate correlation per country
        indicator_correlations = {indicator: [] for indicator in indicators}

        for country in data['Country Name'].unique():
            country_data = data[data['Country Name'] == country]
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
            
        # Heatmap
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
        st.error(f"Error in correlation analysis: {e}")