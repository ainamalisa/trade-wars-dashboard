import streamlit as st

def show():
    # --- SECTION 7: Conclusion & Recommendations ---
    st.header("Conclusion & Recommendations")
    with st.expander("1. Trade Volume Impact"):
        st.markdown("""
        - Tariffs had a **long-lasting dampening effect** on electronics trade.
        - Trade patterns remain **below pre-COVID levels**, despite temporary rebounds.
        """)

    with st.expander("2. Economic Ripple Effects"):
        st.markdown("""
        - Tariff-induced shocks extended **beyond the US and China**.
        - Consequences include **inflation, reduced GDP growth, and employment pressure** in third-party countries.
        """)

    with st.expander("3. Public Sentiment Trends"):
        st.markdown("""
        - News sentiment analysis revealed **growing negativity and fatigue**.
        - Key triggers included **inflation spikes** and **tariff expansions**.
        """)

    with st.expander("4. Predictive Model Confirmation"):
        st.markdown("""
        - **Machine learning models (e.g., Random Forest)** confirm **trade volume decline** under tariff escalation.
        - Spillover impacts observed across multiple global economies.
        """)

    # Divider
    st.markdown("---")

    # Final Policy Recommendations
    st.header("📌 Final Policy Recommendations")

    with st.expander("1. Risk Management"):
        st.markdown("""
        - Prepare for **up to 30% trade volume decline** under **worst-case scenarios**.
        - Implement **resilience plans** across industries and sectors.
        """)

    with st.expander("2. Diversification Strategy"):
        st.markdown("""
        - **Reduce overreliance on China** for critical imports, especially in **electronics**.
        - Explore **alternative trade partners and regional supply chains**.
        """)

    with st.expander("3. Sentiment Monitoring"):
        st.markdown("""
        - Track **public/media sentiment** using tools like **VADER or NLP APIs**.
        - Use as early indicators of **political shifts and consumer behavior**.
        """)

    with st.expander("4. Evidence-Based Trade Policy"):
        st.markdown("""
        - Utilize **AI forecasting models** (e.g., ARIMA, Random Forest) for **dynamic policy design**.
        - Avoid blanket tariffs—prefer **targeted, data-backed interventions**.
        """)