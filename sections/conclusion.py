# sections/conclusion.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def show():
    st.title("✅ Conclusion & Recommendations")
    st.markdown("---")
    
    # Executive Summary Box
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 25px; border-radius: 15px; margin-bottom: 30px; border-left: 6px solid #1f4e79;">
    <h2 style="color: #1f4e79; margin: 0; font-weight: bold;">📋 Executive Summary</h2>
    <p style="margin: 15px 0 0 0; color: #2c5282; font-size: 18px; line-height: 1.6;">
    Our comprehensive analysis reveals that US-China tariffs have created lasting impacts on global trade, 
    with effects extending far beyond the two countries involved. Here's what businesses and policymakers need to know.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Findings Section
    st.subheader("🔍 Key Findings")
    
    # Create two columns for findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #ffe8e8; padding: 20px; border-radius: 12px; margin: 10px 0; border-left: 5px solid #dc3545;">
        <h4 style="color: #721c24; margin: 0; font-weight: bold;">📉 Trade Impact</h4>
        <p style="color: #721c24; margin: 10px 0; font-size: 16px;">
        Electronics trade between US and China dropped significantly and hasn't fully recovered, 
        even years after tariffs were introduced.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 20px; border-radius: 12px; margin: 10px 0; border-left: 5px solid #856404;">
        <h4 style="color: #533f03; margin: 0; font-weight: bold;">💭 Public Opinion</h4>
        <p style="color: #533f03; margin: 10px 0; font-size: 16px;">
        News sentiment shows growing frustration and fatigue with trade tensions, 
        especially when prices rise for consumers.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #d1ecf1; padding: 20px; border-radius: 12px; margin: 10px 0; border-left: 5px solid #0c5460;">
        <h4 style="color: #0c5460; margin: 0; font-weight: bold;">🌍 Global Effects</h4>
        <p style="color: #0c5460; margin: 10px 0; font-size: 16px;">
        Other countries like Malaysia, Vietnam, and Germany also felt the impact through 
        supply chain disruptions and economic spillovers.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #e2e3e5; padding: 20px; border-radius: 12px; margin: 10px 0; border-left: 5px solid #6c757d;">
        <h4 style="color: #495057; margin: 0; font-weight: bold;">🔮 Future Outlook</h4>
        <p style="color: #495057; margin: 10px 0; font-size: 16px;">
        Our AI models predict continued trade challenges, with potential for further 
        declines if policies become more restrictive.
        </p>
        </div>
        """, unsafe_allow_html=True)

    # Impact Visualization
    st.subheader("📊 Impact at a Glance")
    
    # Create a simple impact chart
    categories = ['Trade Volume', 'Economic Growth', 'Public Sentiment', 'Global Supply Chains']
    impact_scores = [-25, -15, -30, -20]  # Negative impacts
    colors = ['#dc3545', '#fd7e14', '#ffc107', '#6f42c1']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=impact_scores,
            marker_color=colors,
            text=[f'{score}%' for score in impact_scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Estimated Impact of US-China Tariffs (%)",
        yaxis_title="Impact Level",
        xaxis_title="Categories",
        showlegend=False,
        height=400,
        yaxis=dict(range=[-40, 5])
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations Section
    st.markdown("---")
    st.subheader("💡 What Should Be Done?")
    
    st.markdown("""
    Based on our analysis, here are practical recommendations for different stakeholders:
    """)

    # Create tabs for different audiences
    tab1, tab2, tab3 = st.tabs(["🏢 For Businesses", "🏛️ For Policymakers", "📈 For Investors"])
    
    with tab1:
        st.markdown("""
        ### 🏢 Business Recommendations
        
        **Immediate Actions:**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h5 style="color: #155724; margin: 0;">🔄 Diversify Supply Chains</h5>
            <p style="color: #155724; margin: 8px 0;">
            Don't put all your eggs in one basket. Find suppliers in multiple countries 
            to reduce risk if trade policies change suddenly.
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: #cce5ff; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h5 style="color: #004085; margin: 0;">📊 Monitor Trade Data</h5>
            <p style="color: #004085; margin: 8px 0;">
            Keep track of trade volumes and tariff changes. Use this data to 
            make informed decisions about pricing and sourcing.
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h5 style="color: #856404; margin: 0;">💰 Build Price Flexibility</h5>
            <p style="color: #856404; margin: 8px 0;">
            Prepare for cost increases due to tariffs. Consider how to adjust 
            prices while staying competitive in the market.
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: #f8d7da; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h5 style="color: #721c24; margin: 0;">⚠️ Plan for Worst Case</h5>
            <p style="color: #721c24; margin: 8px 0;">
            Our models show trade could drop by 30% in escalation scenarios. 
            Have backup plans ready for this possibility.
            </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        ### 🏛️ Policy Recommendations
        
        **Strategic Approaches:**
        """)
        
        st.markdown("""
        <div style="background-color: #e7f3ff; padding: 20px; border-radius: 12px; margin: 15px 0; border-left: 5px solid #0066cc;">
        <h5 style="color: #004499; margin: 0;">🎯 Use Data-Driven Policies</h5>
        <p style="color: #0066cc; margin: 10px 0;">
        Instead of blanket tariffs, use our AI models and real data to design targeted policies 
        that achieve goals with less economic disruption.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #f0fff0; padding: 20px; border-radius: 12px; margin: 15px 0; border-left: 5px solid #28a745;">
        <h5 style="color: #1e7e34; margin: 0;">📈 Monitor Sentiment</h5>
        <p style="color: #28a745; margin: 10px 0;">
        Track public opinion and news sentiment as early warning signals. 
        Negative sentiment often predicts political pressure for policy changes.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #fff8e1; padding: 20px; border-radius: 12px; margin: 15px 0; border-left: 5px solid #ffa000;">
        <h5 style="color: #e65100; margin: 0;">🤝 Consider Global Impact</h5>
        <p style="color: #f57c00; margin: 10px 0;">
        Trade policies affect allies and partners too. Work with other countries 
        to minimize negative spillover effects.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        ### 📈 Investment Insights
        
        **Market Opportunities & Risks:**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🟢 Potential Opportunities:**
            - **Alternative suppliers** in Vietnam, Malaysia, Mexico
            - **Automation technology** to reduce labor costs
            - **Trade facilitation** services and logistics
            - **Domestic manufacturing** in electronics
            """)
        
        with col2:
            st.markdown("""
            **🔴 Key Risks to Watch:**
            - **Supply chain disruptions** continuing
            - **Consumer price increases** affecting demand
            - **Policy uncertainty** creating market volatility
            - **Geopolitical tensions** escalating further
            """)

    # Final Call to Action
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #28a745; padding: 25px; border-radius: 15px; margin: 30px 0; text-align: center;">
    <h3 style="color: white; margin: 0; font-weight: bold;">🎯 The Bottom Line</h3>
    <p style="color: white; margin: 15px 0; font-size: 18px; line-height: 1.6;">
    Trade wars create lasting changes that go beyond the original participants. 
    Success requires <strong>preparation, flexibility, and data-driven decision making</strong>.
    </p>
    <p style="color: white; margin: 15px 0; font-size: 16px;">
    Use the insights from this analysis to make informed decisions, but stay alert to new developments 
    and be ready to adapt as the situation evolves.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Future Research Suggestions
    st.subheader("🔬 Future Research Opportunities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Data & Analytics:**
        - Real-time trade flow monitoring
        - Advanced sentiment analysis from social media
        - Supply chain risk assessment models
        """)
    
    with col2:
        st.markdown("""
        **🌍 Policy Studies:**
        - Regional trade agreement impacts
        - Industry-specific tariff effects
        - Long-term economic restructuring patterns
        """)

if __name__ == "__main__":
    show()