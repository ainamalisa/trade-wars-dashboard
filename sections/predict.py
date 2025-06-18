# predict.py - Predictive Modeling Section for TradeWars Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def show():
    st.title("📉 Predictive Modeling: Future Trade Impact")
    
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1f4e79;">
    <h3 style="color: #1f4e79; margin: 0; font-weight: bold;">🎯 What We're Predicting</h3>
    <p style="margin: 15px 0 0 0; color: #2c5282; font-size: 16px; line-height: 1.5;">Using advanced machine learning models, we forecast how US-China electronics trade will evolve through 2027 under different policy scenarios.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and prepare data
    try:
        # Load the preprocessed data files
        ts_forecasts = pd.read_csv('data/Result/time_series_forecasts.csv')
        master_data = pd.read_csv('data/Result/master_modeling_dataset.csv')
        economic_impact = pd.read_csv('data/Result/economic_impact_summary.csv')
        mc_status_quo = pd.read_csv('data/Result/multicountry_predictions_status_quo.csv')
        mc_escalation = pd.read_csv('data/Result/multicountry_predictions_tariff_escalation.csv')
        
        # Clean the data
        ts_forecasts = ts_forecasts.dropna()
        economic_impact = economic_impact.dropna()
        
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Please ensure all model result files are available in the project directory.")
        return
    
    # Create tabs for different aspects of predictions
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Future Forecasts", 
        "🌍 Scenario Analysis", 
        "📊 Economic Impact", 
        "🎯 Model Performance"
    ])
    
    with tab1:
        show_future_forecasts(ts_forecasts, master_data)
    
    with tab2:
        show_scenario_analysis(mc_status_quo, mc_escalation)
    
    with tab3:
        show_economic_impact(economic_impact)
    
    with tab4:
        show_model_performance(master_data)

def show_future_forecasts(ts_forecasts, master_data):
    """Display main trade forecast predictions"""
    
    st.subheader("📈 US-China Electronics Trade Forecasts (2025-2027)")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("""
        ### 📋 Key Predictions
        
        **What this shows:**
        - Expected trade values for next 3 years
        - Confidence intervals (uncertainty ranges)
        - Comparison with historical trends
        
        **How to read:**
        - 📈 **Blue line**: Historical actual trade
        - 🔴 **Red line**: AI predictions
        - 🌫️ **Gray area**: Uncertainty range
        """)
        
        # Display key numbers
        if len(ts_forecasts) > 0:
            avg_forecast = ts_forecasts['forecast'].mean() / 1e9  # Convert to billions
            st.metric(
                "Average Annual Trade", 
                f"${avg_forecast:.1f}B",
                help="Expected average trade value 2025-2027"
            )
            
            # Calculate trend
            if len(ts_forecasts) >= 2:
                trend = ((ts_forecasts['forecast'].iloc[-1] - ts_forecasts['forecast'].iloc[0]) / 
                        ts_forecasts['forecast'].iloc[0] * 100)
                st.metric(
                    "3-Year Trend", 
                    f"{trend:+.1f}%",
                    help="Expected growth from 2025 to 2027"
                )
    
    with col1:
        # Create the main forecast chart
        fig = go.Figure()
        
        # Add historical data
        if len(master_data) > 0:
            fig.add_trace(go.Scatter(
                x=master_data['year'],
                y=master_data['imports_from_china'] / 1e9,
                mode='lines+markers',
                name='📈 Historical Trade',
                line=dict(color='steelblue', width=3),
                marker=dict(size=8)
            ))
        
        # Add forecast data
        if len(ts_forecasts) > 0:
            fig.add_trace(go.Scatter(
                x=ts_forecasts['year'],
                y=ts_forecasts['forecast'] / 1e9,
                mode='lines+markers',
                name='🔮 AI Forecast',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond')
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=list(ts_forecasts['year']) + list(ts_forecasts['year'][::-1]),
                y=list(ts_forecasts['upper_ci'] / 1e9) + list(ts_forecasts['lower_ci'][::-1] / 1e9),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='🌫️ Uncertainty Range',
                hoverinfo='skip'
            ))
        
        # Add vertical line to separate historical from forecast
        if len(master_data) > 0 and len(ts_forecasts) > 0:
            last_historical_year = master_data['year'].max()
            fig.add_vline(
                x=last_historical_year + 0.5,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Starts",
                annotation_position="top"
            )
        
        fig.update_layout(
            title="US Electronics Imports from China: Historical vs Predicted",
            xaxis_title="Year",
            yaxis_title="Trade Value (Billions USD)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    ### 💡 What This Means
    
    **For Businesses:**
    - Plan inventory and supply chains based on expected trade volumes
    - Prepare for potential fluctuations within the uncertainty range
    
    **For Policymakers:**
    - Understand the baseline trajectory of US-China trade
    - Use forecasts to evaluate policy impact potential
    
    **For Investors:**
    - Assess market size trends in US-China electronics trade
    - Consider uncertainty ranges for risk management
    """)

def show_scenario_analysis(mc_status_quo, mc_escalation):
    """Display different policy scenario comparisons"""
    
    st.subheader("🌍 Policy Scenario Analysis")
    
    st.markdown("""
    ### 🎭 Two Possible Futures
    
    We modeled two main scenarios to understand how different policies might affect trade:
    """)
    
    # Create scenario selector
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #2d5a2d;">
        <h4 style="color: #1a4d1a; margin: 0; font-weight: bold;">✅ Status Quo Scenario</h4>
        <p style="color: #2d5a2d; margin: 10px 0; font-weight: 600;">What happens: Current policies continue</p>
        <ul style="color: #2d5a2d; margin: 0; padding-left: 20px;">
        <li style="margin: 5px 0;">No major tariff changes</li>
        <li style="margin: 5px 0;">Existing trade agreements remain</li>
        <li style="margin: 5px 0;">Normal diplomatic relations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #ffe8e8; padding: 20px; border-radius: 10px; border-left: 5px solid #8b2635;">
        <h4 style="color: #8b2635; margin: 0; font-weight: bold;">⚠️ Tariff Escalation Scenario</h4>
        <p style="color: #8b2635; margin: 10px 0; font-weight: 600;">What happens: Trade tensions increase</p>
        <ul style="color: #8b2635; margin: 0; padding-left: 20px;">
        <li style="margin: 5px 0;">50% increase in tariff rates</li>
        <li style="margin: 5px 0;">More restrictive trade policies</li>
        <li style="margin: 5px 0;">Heightened economic tensions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Filter for US-China trade specifically
    us_china_status_quo = mc_status_quo[
        ((mc_status_quo['country1'] == 'United States') & (mc_status_quo['country2'] == 'China')) |
        ((mc_status_quo['country1'] == 'China') & (mc_status_quo['country2'] == 'United States'))
    ]
    
    us_china_escalation = mc_escalation[
        ((mc_escalation['country1'] == 'United States') & (mc_escalation['country2'] == 'China')) |
        ((mc_escalation['country1'] == 'China') & (mc_escalation['country2'] == 'United States'))
    ]
    
    # Create comparison chart
    fig = go.Figure()
    
    if len(us_china_status_quo) > 0:
        fig.add_trace(go.Scatter(
            x=us_china_status_quo['year'],
            y=us_china_status_quo['predicted_trade_intensity'],
            mode='lines+markers',
            name='✅ Status Quo',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
    
    if len(us_china_escalation) > 0:
        fig.add_trace(go.Scatter(
            x=us_china_escalation['year'],
            y=us_china_escalation['predicted_trade_intensity'],
            mode='lines+markers',
            name='⚠️ Tariff Escalation',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="US-China Trade Intensity: Scenario Comparison",
        xaxis_title="Year",
        yaxis_title="Trade Intensity Index",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display the impact
    if len(us_china_status_quo) > 0 and len(us_china_escalation) > 0:
        status_quo_avg = us_china_status_quo['predicted_trade_intensity'].mean()
        escalation_avg = us_china_escalation['predicted_trade_intensity'].mean()
        impact_percent = ((escalation_avg - status_quo_avg) / status_quo_avg) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Status Quo Average",
                f"{status_quo_avg:.1f}",
                help="Average trade intensity under current policies"
            )
        
        with col2:
            st.metric(
                "Escalation Average",
                f"{escalation_avg:.1f}",
                help="Average trade intensity under tariff escalation"
            )
        
        with col3:
            st.metric(
                "Impact of Escalation",
                f"{impact_percent:+.1f}%",
                delta=f"{impact_percent:+.1f}%",
                help="Percentage change in trade intensity"
            )
    
    # Add insights
    st.markdown("""
    ### 🔍 Key Insights
    
    **The Trade-off:**
    - **Higher tariffs** may protect domestic industries but **reduce overall trade**
    - **Status quo** maintains current trade levels but **doesn't address trade imbalances**
    
    **For Decision Makers:**
    - Consider the **economic costs** of each scenario
    - Plan for **adjustment periods** regardless of policy direction
    - Monitor **early indicators** to predict which scenario is emerging
    """)

def show_economic_impact(economic_impact):
    """Display broader economic impact predictions"""
    
    st.subheader("📊 Broader Economic Impact")
    
    st.markdown("""
    ### 🏭 Beyond Trade: Effects on the Economy
    
    Tariff changes don't just affect trade - they ripple through the entire economy, 
    affecting growth, employment, and prices.
    """)
    
    if len(economic_impact) > 0:
        # Create metrics for each scenario
        scenarios = economic_impact['Scenario'].unique()
        
        # Display scenario comparison
        cols = st.columns(len(scenarios))
        
        for i, scenario in enumerate(scenarios):
            scenario_data = economic_impact[economic_impact['Scenario'] == scenario].iloc[0]
            
            with cols[i]:
                # Choose color based on scenario
                if 'Baseline' in scenario:
                    color = "blue"
                elif 'Increase' in scenario:
                    color = "red"
                else:
                    color = "green"
                
                st.markdown(f"""
                <div style="background-color: #{color}15; padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: {color}; margin: 0;">{scenario}</h4>
                <p style="margin: 5px 0;"><strong>Tariff Change:</strong> {scenario_data['Tariff_Change']:+.0f} points</p>
                <p style="margin: 5px 0;"><strong>GDP Growth:</strong> {scenario_data['Avg_GDP_Growth']:.1f}%</p>
                <p style="margin: 5px 0;"><strong>Unemployment:</strong> {scenario_data['Avg_Unemployment']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create comparison charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("GDP Growth Impact", "Unemployment Impact"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # GDP Growth chart
        fig.add_trace(
            go.Bar(
                x=economic_impact['Scenario'],
                y=economic_impact['Avg_GDP_Growth'],
                name='GDP Growth (%)',
                marker_color=['blue' if 'Baseline' in x else 'red' if 'Increase' in x else 'green' 
                             for x in economic_impact['Scenario']]
            ),
            row=1, col=1
        )
        
        # Unemployment chart
        fig.add_trace(
            go.Bar(
                x=economic_impact['Scenario'],
                y=economic_impact['Avg_Unemployment'],
                name='Unemployment (%)',
                marker_color=['blue' if 'Baseline' in x else 'red' if 'Increase' in x else 'green' 
                             for x in economic_impact['Scenario']],
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Economic Impact by Scenario",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add economic interpretation
    st.markdown("""
    ### 💰 Economic Trade-offs Explained
    
    **What These Numbers Mean:**
    
    - **GDP Growth**: How fast the economy expands each year
      - Higher = More prosperity, more jobs, higher incomes
      - Lower = Economic slowdown, less business investment
    
    - **Unemployment**: Percentage of people looking for work but can't find jobs
      - Higher = More people out of work, social challenges
      - Lower = More people employed, stronger consumer spending
    
    **The Policy Dilemma:**
    - **Protecting industries** with tariffs may save some jobs
    - But it might **hurt overall economic growth**
    - **Lower tariffs** boost growth but may cost jobs in protected sectors
    """)
    
    # Add policy recommendations
    st.markdown("""
    ### 🎯 What This Means for Policy
    
    **For Policymakers:**
    1. **Gradual Changes**: Avoid sudden policy shifts that shock the economy
    2. **Support Transitions**: Help workers and companies adapt to changes
    3. **Monitor Effects**: Track real economic data to adjust policies
    
    **For Businesses:**
    1. **Diversify**: Don't depend too heavily on any one trade relationship
    2. **Plan Ahead**: Use these forecasts to prepare for different scenarios
    3. **Stay Flexible**: Be ready to adapt as policies change
    """)

def show_model_performance(master_data):
    """Display information about model accuracy and limitations"""
    
    st.subheader("🎯 How Accurate Are Our Predictions?")
    
    st.markdown("""
    ### 🔬 The Science Behind the Forecasts
    
    We used multiple AI models to make these predictions. Here's how we did it and how confident we can be.
    """)
    
    # Model explanation tabs
    model_tab1, model_tab2, model_tab3 = st.tabs([
        "🤖 AI Models Used", 
        "📊 Data Quality", 
        "⚠️ Limitations"
    ])
    
    with model_tab1:
        st.markdown("""
        ### 🧠 Four Different AI Approaches
        
        We combined multiple AI models to get the best predictions:
        
        **1. 📈 Time Series Model (ARIMA)**
        - **What it does**: Analyzes historical trade patterns
        - **Strength**: Good at detecting trends and cycles
        - **Best for**: Short-term forecasts based on past data
        
        **2. 🌍 Multi-Country Model (Random Forest)**
        - **What it does**: Compares trade between 6 different countries
        - **Strength**: Understands how countries affect each other
        - **Best for**: Policy scenario analysis
        
        **3. 💰 Economic Impact Model (Machine Learning)**
        - **What it does**: Predicts effects on GDP and unemployment
        - **Strength**: Shows broader economic consequences
        - **Best for**: Understanding economic trade-offs
        
        **4. 🎯 Ensemble Model (Combined)**
        - **What it does**: Combines all other models with expert weighting
        - **Strength**: Most comprehensive and balanced predictions
        - **Best for**: Final decision-making forecasts
        """)
        
        # Show model confidence scores
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Combined", "4", help="Different AI approaches integrated")
        
        with col2:
            st.metric("Years of Data", f"{len(master_data)}", help="Historical years used for training")
        
        with col3:
            st.metric("Confidence Level", "Medium", help="Overall reliability assessment")
    
    with model_tab2:
        st.markdown("""
        ### 📊 Data Quality Assessment
        
        **What Data We Used:**
        - ✅ **Official Trade Statistics**: US Census Bureau, UN Comtrade
        - ✅ **Economic Indicators**: World Bank, Federal Reserve
        - ✅ **News Sentiment**: 968 news articles analyzed
        - ✅ **Policy Data**: Tariff rates, trade agreements
        """)
        
        if len(master_data) > 0:
            # Show data completeness
            st.markdown("**Data Completeness by Year:**")
            
            # Create a completeness visualization
            completeness_data = []
            for year in master_data['year']:
                year_row = master_data[master_data['year'] == year].iloc[0]
                non_null_count = year_row.notna().sum()
                total_fields = len(year_row)
                completeness = (non_null_count / total_fields) * 100
                
                completeness_data.append({
                    'Year': year,
                    'Completeness': completeness
                })
            
            completeness_df = pd.DataFrame(completeness_data)
            
            fig = px.bar(
                completeness_df, 
                x='Year', 
                y='Completeness',
                title="Data Completeness by Year (%)",
                color='Completeness',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Strengths:**
        - High-quality official government data
        - Multiple independent data sources
        - Real-time sentiment analysis integration
        
        **Challenges:**
        - Limited historical data (only 7 years)
        - COVID-19 created unusual patterns
        - Some economic indicators have delays
        """)
    
    with model_tab3:
        st.markdown("""
        ### ⚠️ Important Limitations to Consider
        
        **Why You Should Be Cautious:**
        
        **1. 📅 Limited History**
        - Only 7 years of data available
        - Trade wars are relatively new phenomenon
        - Hard to predict unprecedented situations
        
        **2. 🌍 External Shocks**
        - COVID-19 disrupted normal patterns
        - Geopolitical events can change everything quickly
        - Natural disasters, wars, pandemics not predictable
        
        **3. 🏛️ Policy Changes**
        - New governments may completely change policies
        - International agreements can shift rapidly
        - Models assume current policy frameworks continue
        
        **4. 📊 Data Delays**
        - Official trade data takes months to publish
        - Economic indicators are often revised
        - Real-time effects may differ from predictions
        """)
        
        # Uncertainty ranges
        st.markdown("""
        ### 🎯 How to Use These Predictions
        
        **✅ Good for:**
        - Understanding general trends and directions
        - Comparing different policy scenarios
        - Planning for multiple possible futures
        - Making informed business and policy decisions
        
        **❌ Don't use for:**
        - Exact predictions of specific trade values
        - Making decisions without considering alternatives
        - Ignoring real-world developments as they happen
        - Assuming the future will definitely match predictions
        
        **💡 Best Practice:**
        - Use as **guidance**, not **gospel**
        - **Monitor real data** and adjust as needed
        - **Prepare for multiple scenarios**
        - **Update predictions** as new data becomes available
        """)
    
    # Final summary
    st.markdown("""
    ---
    ### 🎯 Bottom Line
    
    Our AI models provide **valuable insights** into possible futures for US-China trade, 
    but they're **tools for better decision-making**, not crystal balls. 
    
    Use them to **prepare for different scenarios** and **understand potential impacts**, 
    but always **stay flexible** and **monitor real-world developments**.
    """)

if __name__ == "__main__":
    show()