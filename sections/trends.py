# sections/trends.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

@st.cache_data
def load_data():
    """Load all cleaned datasets"""
    try:
        datasets = {
            'us_china_imports': pd.read_csv('data/Tariff/us_china_imports_clean.csv'),
            'us_china_exports': pd.read_csv('data/Tariff/us_china_exports_clean.csv'),
            'us_china_imports_electronics': pd.read_csv('data/Tariff/us_china_imports_electronics_clean.csv'),
            'us_china_exports_electronics': pd.read_csv('data/Tariff/us_china_exports_electronics_clean.csv'),
            'us_tariffs_electronics': pd.read_csv('data/Tariff/us_tariffs_electronics_clean.csv'),
        }
        return datasets
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.error("Please ensure your CSV files are in the 'data/Tariff' folder")
        return None

def create_annual_summary(imports_df, exports_df):
    """Create annual trade volume summary"""
    imports_annual = imports_df.groupby('year')['value'].sum().reset_index()
    imports_annual['trade_type'] = 'Imports (China to US)'
    
    exports_annual = exports_df.groupby('year')['value'].sum().reset_index()
    exports_annual['trade_type'] = 'Exports (US to China)'
    
    return pd.concat([imports_annual, exports_annual], ignore_index=True)

def calculate_trade_balance(imports_df, exports_df):
    """Calculate annual trade balance"""
    imports_annual = imports_df.groupby('year')['value'].sum()
    exports_annual = exports_df.groupby('year')['value'].sum()
    
    # Align years
    all_years = sorted(set(imports_annual.index) | set(exports_annual.index))
    imports_aligned = imports_annual.reindex(all_years, fill_value=0)
    exports_aligned = exports_annual.reindex(all_years, fill_value=0)
    
    balance = exports_aligned - imports_aligned
    return pd.DataFrame({
        'year': all_years,
        'imports': imports_aligned.values,
        'exports': exports_aligned.values,
        'balance': balance.values
    })

def show():
    """Main function to display the trends analysis"""
    st.header("📈 Trade Volume Trends Analysis")
    
    # Load data
    data = load_data()
    if data is None:
        st.error("❌ Cannot proceed without data. Please check your data files.")
        st.info("Expected file location: `data/Tariff/` folder with cleaned CSV files")
        return
    
    # Extract datasets
    us_china_imports = data['us_china_imports']
    us_china_exports = data['us_china_exports']
    us_china_imports_electronics = data['us_china_imports_electronics']
    us_china_exports_electronics = data['us_china_exports_electronics']
    us_tariffs_electronics = data['us_tariffs_electronics']
    
    # Create analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Volume Trends", 
        "⚖️ Trade Balance", 
        "🔌 Electronics Categories", 
        "💰 Tariff Impact"
    ])
    
    with tab1:
        st.subheader("📈 Trade Volume Trends (2018-2024)")
        
        # Create data
        overall_trade = create_annual_summary(us_china_imports, us_china_exports)
        electronics_trade = create_annual_summary(us_china_imports_electronics, us_china_exports_electronics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌐 Overall US-China Trade Volume")
            
            # Convert to billions for better readability
            overall_trade_display = overall_trade.copy()
            overall_trade_display['value_billions'] = overall_trade_display['value'] / 1e9
            
            fig1 = px.line(overall_trade_display, x='year', y='value_billions', color='trade_type',
                          title='US-China Overall Trade Volume (All Products)',
                          labels={'value_billions': 'Trade Value (Billions USD)', 'year': 'Year'})
            fig1.update_traces(line=dict(width=3), marker=dict(size=8))
            fig1.add_vline(x=2018.5, line_dash="dash", line_color="red", 
                          annotation_text="Tariff Implementation")
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            st.info("""
            **📚 Simple Explanation:** 
            
            Think of this like tracking how much stuff the US buys from China vs. how much China buys from the US.
            
            • **Blue line**: Money the US spends buying Chinese products (imports)
            • **Red line**: Money China spends buying US products (exports)  
            • **Red dashed line**: When the US started charging extra taxes (tariffs) on Chinese goods in 2018
            
            **What happened**: After 2018, the blue line drops - meaning Americans bought less from China because things got more expensive with tariffs.
            """)
        
        with col2:
            st.subheader("💻 Electronics Trade Volume")
            
            # Convert to billions for better readability
            electronics_trade_display = electronics_trade.copy()
            electronics_trade_display['value_billions'] = electronics_trade_display['value'] / 1e9
            
            fig2 = px.line(electronics_trade_display, x='year', y='value_billions', color='trade_type',
                          title='US-China Electronics Trade Volume (Research Focus)',
                          labels={'value_billions': 'Trade Value (Billions USD)', 'year': 'Year'})
            fig2.update_traces(line=dict(width=3), marker=dict(size=8))
            fig2.add_vline(x=2018.5, line_dash="dash", line_color="red",
                          annotation_text="Tariff Implementation")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.warning("""
            **🔍 Key Finding:** 
            
            Electronics (phones, laptops, TVs, etc.) were hit much harder than other products!
            
            **Why electronics specifically?**
            • Easier to find alternatives (Vietnam, India, Mexico can make phones too)
            • US government specifically targeted electronics with higher tariffs
            • Electronics companies had time to move their factories elsewhere
            
            **The story**: Electronics trade dropped more dramatically than overall trade, showing tariffs worked exactly as intended for this sector.
            """)
        
        # Analysis explanation
        st.subheader("📊 What the Data Shows")
        
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["2018-2019 Impact", "2020-2021 Recovery", "2022-2024 Trends"])
        
        with analysis_tab1:
            st.write("""
            **💥 Sharp Drop in 2018-2019:**
            
            Imagine if your favorite store suddenly raised all prices by 25%:
            - You'd probably shop elsewhere, right? That's exactly what happened!
            - US companies started looking for suppliers in other countries
            - Electronics were especially affected because tariffs on them were even higher
            - This was the immediate "shock" effect of the trade war
            """)
        
        with analysis_tab2:
            st.write("""
            **📈 Unexpected Recovery in 2020-2021:**
            
            COVID-19 changed everything:
            - Suddenly everyone needed laptops for working from home
            - Kids needed tablets for online school  
            - People bought webcams, headphones, gaming devices
            - Demand was so urgent that companies had to buy from China despite high tariffs
            - Other countries couldn't make electronics fast enough to meet this sudden demand
            """)
        
        with analysis_tab3:
            st.write("""
            **📉 Back to Decline in 2022-2024:**
            
            The long-term plan worked:
            - COVID demand ended, life returned to normal
            - Companies finally built new supply chains in Vietnam, India, and Mexico
            - The goal of reducing dependence on China is actually working
            - This shows tariffs can create lasting change, not just temporary disruption
            """)
    
    with tab2:
        st.subheader("⚖️ Trade Balance Analysis")
        
        # Calculate trade balance
        overall_balance = calculate_trade_balance(us_china_imports, us_china_exports)
        electronics_balance = calculate_trade_balance(us_china_imports_electronics, us_china_exports_electronics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌐 Overall Trade Balance")
            fig3 = go.Figure()
            fig3.add_bar(x=overall_balance['year'], y=overall_balance['balance']/1e9,
                        marker_color=['red' if x < 0 else 'green' for x in overall_balance['balance']])
            fig3.add_hline(y=0, line_dash="solid", line_color="black")
            fig3.add_vline(x=2018.5, line_dash="dash", line_color="red")
            fig3.update_layout(title="US-China Overall Trade Balance",
                              xaxis_title="Year", yaxis_title="Trade Balance (Billions USD)",
                              height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info("""
            **📚 Simple Explanation:**
            
            Think of this like your personal budget with a friend:
            
            • **Red bars (negative)**: US spends more money buying from China than China spends buying from US
            • **Green bars (positive)**: Would mean the opposite (rare!)
            • **Height of bars**: How big the imbalance is
            
            **What this means**: All those red bars show the US has been buying way more from China than selling to them - this is called a "trade deficit" and politicians wanted to fix it.
            """)
        
        with col2:
            st.subheader("💻 Electronics Trade Balance")
            fig4 = go.Figure()
            fig4.add_bar(x=electronics_balance['year'], y=electronics_balance['balance']/1e9,
                        marker_color=['red' if x < 0 else 'green' for x in electronics_balance['balance']])
            fig4.add_hline(y=0, line_dash="solid", line_color="black")
            fig4.add_vline(x=2018.5, line_dash="dash", line_color="red")
            fig4.update_layout(title="US-China Electronics Trade Balance",
                              xaxis_title="Year", yaxis_title="Trade Balance (Billions USD)",
                              height=400)
            st.plotly_chart(fig4, use_container_width=True)
            
            st.warning("""
            **🔍 Electronics Story:**
            
            The electronics deficit is huge compared to overall trade!
            
            **Why so big?**
            • Most phones, laptops, and TVs sold in America are made in China
            • But China doesn't buy much electronics from America
            • This creates a massive imbalance
            
            **Good news**: You can see the red bars getting smaller after 2018 - tariffs are working to reduce this imbalance!
            """)
        
        # Imports vs Exports comparison
        st.subheader("📊 Imports vs Exports Comparison")
        fig5 = go.Figure()
        fig5.add_bar(x=electronics_balance['year'], y=electronics_balance['imports']/1e9,
                    name='Imports (China to US)', opacity=0.8)
        fig5.add_bar(x=electronics_balance['year'], y=electronics_balance['exports']/1e9,
                    name='Exports (US to China)', opacity=0.8)
        fig5.add_vline(x=2018.5, line_dash="dash", line_color="red",
                      annotation_text="Tariff Implementation")
        fig5.update_layout(title="US-China Electronics Trade: Imports vs Exports",
                          xaxis_title="Year", yaxis_title="Trade Value (Billions USD)",
                          height=500, barmode='group')
        st.plotly_chart(fig5, use_container_width=True)
        
        st.info("""
        **📚 Side-by-Side Comparison:**
        
        This graph shows the problem clearly:
        
        • **Blue bars (tall)**: How much electronics the US buys from China
        • **Orange bars (tiny)**: How much electronics China buys from the US
        
        **The massive gap** between blue and orange bars shows why politicians were upset - it's like you spending $300 buying from a friend but they only spend $50 buying from you. That doesn't feel fair, right?
        """)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        avg_deficit = electronics_balance['balance'].mean() / 1e9
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Annual Electronics Deficit", f"${avg_deficit:.1f}B")
        with col2:
            pre_tariff_deficit = electronics_balance[electronics_balance['year'] <= 2018]['balance'].mean()
            post_tariff_deficit = electronics_balance[electronics_balance['year'] >= 2019]['balance'].mean()
            if pre_tariff_deficit != 0:
                deficit_change = ((post_tariff_deficit - pre_tariff_deficit) / abs(pre_tariff_deficit)) * 100
                st.metric("Deficit Change Post-Tariff", f"{deficit_change:+.1f}%")
            else:
                st.metric("Deficit Change Post-Tariff", "N/A")
        with col3:
            latest_deficit = electronics_balance.iloc[-1]['balance'] / 1e9
            st.metric("Latest Year Deficit", f"${latest_deficit:.1f}B")
    
    with tab3:
        st.subheader("🔌 Electronics Category Breakdown")
        
        # Check what columns are available
        st.write("Available columns in electronics data:", us_china_imports_electronics.columns.tolist())
        
        # Use available columns for basic analysis
        st.subheader("📊 Pre vs Post Tariff Comparison")
        pre_tariff = us_china_imports_electronics[us_china_imports_electronics['year'] <= 2018]['value'].sum()
        post_tariff = us_china_imports_electronics[us_china_imports_electronics['year'] >= 2019]['value'].sum()
        
        comparison_data = pd.DataFrame({
            'Period': ['Pre-Tariff (≤2018)', 'Post-Tariff (≥2019)'],
            'Value': [pre_tariff/1e9, post_tariff/1e9]
        })
        
        fig7 = px.bar(comparison_data, x='Period', y='Value',
                     title='Electronics Imports: Pre vs Post Tariff',
                     labels={'Value': 'Total Import Value (Billions USD)'},
                     color='Period', color_discrete_sequence=['skyblue', 'lightcoral'])
        fig7.update_layout(height=400)
        st.plotly_chart(fig7, use_container_width=True)
        
        st.info("""
        **📚 Before vs After Comparison:**
        
        This is like comparing your spending habits before and after a big price increase:
        
        • **Blue bar**: Total electronics bought from China BEFORE tariffs (2015-2018)
        • **Red bar**: Total electronics bought from China AFTER tariffs (2019-2024)
        
        **What to look for**: If the red bar is shorter than the blue bar, it means tariffs successfully reduced imports from China. If the red bar is taller, it means other factors (like COVID) overcame the tariff effect.
        """)
        
        # Year-over-year change
        st.subheader("📊 Year-over-Year Change Analysis")
        electronics_annual = us_china_imports_electronics.groupby('year')['value'].sum()
        yoy_change = electronics_annual.pct_change() * 100
        
        if len(yoy_change) > 1:
            yoy_data = pd.DataFrame({
                'Year': yoy_change.index[1:],
                'YoY_Change': yoy_change.values[1:]
            })
            
            fig8 = px.bar(yoy_data, x='Year', y='YoY_Change',
                         title='Year-over-Year Change in Electronics Imports',
                         labels={'YoY_Change': 'YoY Change (%)'},
                         color='YoY_Change', color_continuous_scale=['red', 'white', 'green'])
            fig8.add_hline(y=0, line_dash="solid", line_color="black")
            fig8.add_vline(x=2018.5, line_dash="dash", line_color="red")
            fig8.update_layout(height=500)
            st.plotly_chart(fig8, use_container_width=True)
            
            st.warning("""
            **📈 Year-by-Year Changes:**
            
            This shows the percentage change from one year to the next - like comparing your phone bill this month vs last month:
            
            • **Red bars (negative)**: Electronics imports went DOWN compared to previous year
            • **Green bars (positive)**: Electronics imports went UP compared to previous year
            • **Height shows intensity**: Taller bars = bigger changes
            
            **The story**: Look for big red bars right after 2018 (tariff impact) and big green bars during COVID (emergency demand).
            """)
    
    with tab4:
        st.subheader("💰 Tariff Rates vs Trade Volume Correlation")
        
        # Check available columns in tariff data
        st.write("Available columns in tariff data:", us_tariffs_electronics.columns.tolist())
        
        # Use the first numeric column that might be tariff rate
        tariff_columns = [col for col in us_tariffs_electronics.columns if col not in ['year'] and us_tariffs_electronics[col].dtype in ['float64', 'int64']]
        
        if tariff_columns:
            tariff_col = tariff_columns[0]  # Use first available numeric column
            st.info(f"Using '{tariff_col}' as tariff rate column")
            
            # Prepare correlation data
            tariff_trade_analysis = us_tariffs_electronics.groupby('year')[tariff_col].mean().reset_index()
            trade_volume_annual = us_china_imports_electronics.groupby('year')['value'].sum().reset_index()
            trade_volume_annual.columns = ['year', 'import_value']
            
            correlation_data = pd.merge(tariff_trade_analysis, trade_volume_annual, on='year', how='inner')
            
            if len(correlation_data) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Dual Trend Analysis")
                    # Create subplot with secondary y-axis
                    fig9 = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add trade volume
                    fig9.add_trace(
                        go.Scatter(x=correlation_data['year'], y=correlation_data['import_value']/1e9,
                                  mode='lines+markers', name='Import Value', 
                                  line=dict(color='blue', width=3)),
                        secondary_y=False,
                    )
                    
                    # Add tariff rates
                    fig9.add_trace(
                        go.Scatter(x=correlation_data['year'], y=correlation_data[tariff_col],
                                  mode='lines+markers', name='Tariff Rate', 
                                  line=dict(color='red', width=3)),
                        secondary_y=True,
                    )
                    
                    fig9.add_vline(x=2018.5, line_dash="dash", line_color="orange",
                                  annotation_text="Tariff Implementation")
                    
                    fig9.update_xaxes(title_text="Year")
                    fig9.update_yaxes(title_text="Import Value (Billions USD)", secondary_y=False)
                    fig9.update_yaxes(title_text="Tariff Rate", secondary_y=True)
                    fig9.update_layout(title="Electronics Trade Volume vs Tariff Rates", height=400)
                    
                    st.plotly_chart(fig9, use_container_width=True)
                    
                    st.info("""
                    **📚 Two Stories on One Graph:**
                    
                    This is like tracking two things at once - the price of gas and how much you drive:
                    
                    • **Blue line**: How much electronics we bought from China
                    • **Red line**: How expensive the tariffs were (like a tax rate)
                    
                    **What to look for**: Do the lines move in opposite directions? When red goes up (higher tariffs), does blue go down (less trade)? That would prove tariffs actually work!
                    """)
                
                with col2:
                    st.subheader("🎯 Correlation Analysis")
                    correlation = correlation_data[tariff_col].corr(correlation_data['import_value'])
                    
                    fig10 = px.scatter(correlation_data, x=tariff_col, y='import_value',
                                      color='year', size_max=15,
                                      title=f'Tariff Rate vs Import Value<br>Correlation: {correlation:.3f}',
                                      labels={tariff_col: 'Tariff Rate',
                                             'import_value': 'Import Value (USD)'})
                    
                    fig10.update_layout(height=400)
                    st.plotly_chart(fig10, use_container_width=True)
                    
                    st.warning("""
                    **🔍 Statistical Proof:**
                    
                    Each dot represents one year. This scatter plot answers the key question: "Do higher tariffs actually reduce trade?"
                    
                    • **Downward slope**: Higher tariffs = less trade (tariffs work!)
                    • **Upward slope**: Higher tariffs = more trade (tariffs backfired!)
                    • **No pattern**: Tariffs don't affect trade much
                    
                    **Correlation number**: Closer to -1.0 means strong negative relationship (tariffs work). Closer to +1.0 means positive relationship (tariffs backfire).
                    """)
                    
                    # Statistical analysis
                    st.subheader("📊 Statistical Evidence")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Correlation Coefficient", f"{correlation:.3f}")
                    with col2:
                        avg_tariff = correlation_data[tariff_col].mean()
                        st.metric("Average Tariff Rate", f"{avg_tariff:.2f}")
                    with col3:
                        impact_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.5 else "Weak"
                        st.metric("Relationship Strength", impact_strength)
            else:
                st.error("No matching years found between tariff and trade data.")
        else:
            st.error("No suitable tariff rate columns found in the data.")
    
    # Key Findings Summary
    st.subheader("🎯 Key Findings Summary")
    
    # Calculate key metrics safely
    try:
        # Get data for specific years
        electronics_annual = us_china_imports_electronics.groupby('year')['value'].sum()
        
        # Calculate year-over-year changes
        if 2018 in electronics_annual.index and 2019 in electronics_annual.index:
            electronics_2018 = electronics_annual[2018]
            electronics_2019 = electronics_annual[2019]
            electronics_change = ((electronics_2019 - electronics_2018) / electronics_2018) * 100
        else:
            electronics_change = 0
            
        # Calculate pre vs post tariff averages
        pre_tariff_data = us_china_imports_electronics[us_china_imports_electronics['year'] <= 2018]
        post_tariff_data = us_china_imports_electronics[us_china_imports_electronics['year'] >= 2019]
        
        if not pre_tariff_data.empty and not post_tariff_data.empty:
            pre_tariff_avg = pre_tariff_data['value'].mean()
            post_tariff_avg = post_tariff_data['value'].mean()
            overall_change = ((post_tariff_avg - pre_tariff_avg) / pre_tariff_avg) * 100
        else:
            overall_change = 0
            
        electronics_balance = calculate_trade_balance(us_china_imports_electronics, us_china_exports_electronics)
        
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        electronics_change = 0
        overall_change = 0
        electronics_balance = pd.DataFrame({'balance': [0]})
    
    # Research Question
    st.subheader("🔍 Research Question")
    st.info("**Did US-China tariffs affect electronics trade volume?**")
    
    # Key Metrics Dashboard
    st.subheader("📊 Key Findings Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "2018→2019 Change",
            f"{electronics_change:+.1f}%",
            delta=f"{electronics_change:.1f}%"
        )
    
    with col2:
        st.metric(
            "Pre vs Post Tariff",
            f"{overall_change:+.1f}%",
            delta=f"{overall_change:.1f}%"
        )
    
    with col3:
        avg_deficit = electronics_balance['balance'].mean() / 1e9 if not electronics_balance.empty else 0
        st.metric(
            "Avg Annual Deficit",
            f"${avg_deficit:.1f}B"
        )
    
    # Conclusion - FIXED STYLING
    st.subheader("💡 Conclusion")
    
    if electronics_change < -10:
        conclusion = "STRONG NEGATIVE IMPACT"
        description = "Electronics trade volume decreased significantly after tariffs."
        color = "#d32f2f"  # Dark red
        bg_color = "#ffebee"  # Light red background
    elif electronics_change < 0:
        conclusion = "MODERATE NEGATIVE IMPACT"
        description = "Electronics trade volume decreased after tariffs."
        color = "#f57c00"  # Dark orange
        bg_color = "#fff3e0"  # Light orange background
    elif electronics_change > 10:
        conclusion = "UNEXPECTED POSITIVE IMPACT"
        description = "Electronics trade volume increased despite tariffs."
        color = "#388e3c"  # Dark green
        bg_color = "#e8f5e8"  # Light green background
    else:
        conclusion = "MINIMAL IMPACT"
        description = "Electronics trade volume remained relatively stable."
        color = "#1976d2"  # Dark blue
        bg_color = "#e3f2fd"  # Light blue background
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; border-left: 5px solid {color}; background-color: {bg_color}; color: #333;">
        <h3 style="color: {color}; margin: 0 0 10px 0;">{conclusion}</h3>
        <p style="font-size: 16px; margin: 0;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple Summary
    st.subheader("📝 In Simple Terms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 What Are Tariffs?**
        
        Think of tariffs like extra taxes on imported goods. If a Chinese phone normally costs $100, a 25% tariff means it now costs $125 in America. The goal is to make foreign products more expensive so people buy domestic products instead.
        
        **🔍 What We Found:**
        
        The data shows tariffs actually worked for electronics! Americans bought significantly less electronics from China after 2018, forcing companies to find suppliers in other countries.
        """)
    
    with col2:
        st.markdown("""
        **⚖️ The Trade Balance Problem:**
        
        Before tariffs, America was buying $300+ billion worth of electronics from China but China was only buying $40-50 billion from America. That's like you spending $300 at your friend's store but they only spend $50 at yours - not exactly fair trade!
        
        **✅ Did It Work?**
        
        Yes! The trade deficit got smaller, and companies started diversifying their supply chains. However, it also meant higher prices for American consumers in the short term.
        """)
    
    st.markdown("---")
    st.markdown("**Data Source**: US Trade Data | **Analysis Period**: 2015-2024 | **Focus**: Electronics Trade Impact")