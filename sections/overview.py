import streamlit as st

def show():
    # --- SECTION 1: Project Overview ---
    st.title("TradeWars: Impact Analysis of US-China Tariffs")
    with st.expander("📘 Brief History of US–China Trade"):
        st.markdown("""
    The United States and China have been key trading partners for decades. US imports from China grew rapidly after China joined the WTO in 2001, making China the largest supplier of goods to the US by 2010. However, concerns over intellectual property theft, trade imbalances, and unfair subsidies led to increasing tensions in the late 2010s.
    """)

    with st.expander("🚫 Timeline of Tariffs and Affected Sectors"):
        st.markdown("""
        In **mid-2018**, the US began imposing tariffs on hundreds of Chinese goods—starting with **steel and aluminum**, then expanding to **electronics, machinery, textiles, and agriculture**. China responded with retaliatory tariffs on US exports.
        
        - **2018**: First wave of tariffs (Section 301)
        - **2019**: Expanded tariffs covering over $350B in goods
        - **2020-2021**: Temporary relief during COVID
        - **2022 onward**: Continued tariffs with targeted exclusions
        """)

    with st.expander("🌎 Why This Issue Matters"):
        st.markdown("""
        Tariffs between the US and China have **global ripple effects**:
        
        - Disruption to **global supply chains**
        - **Higher costs** for consumers and manufacturers
        - Impact on **emerging economies** dependent on trade with China
        - Political tensions influencing global trade alliances

        Understanding these effects helps businesses, policymakers, and researchers make informed decisions.
        """)

    # Use columns for visual layout of objectives
    st.subheader("🎯 Project Objectives")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📉 Economic Impact")
        st.write("Analyze the short- and long-term impact of tariffs on trade volume and economic performance.")

    with col2:
        st.markdown("#### 🏭 Industry Disruption")
        st.write("Explore how tariffs affected specific sectors, especially electronics and manufacturing.")

    with col3:
        st.markdown("#### 🔮 Future Predictions")
        st.write("Use data models to forecast future trends in trade flow, supply chain shifts, and policy impact.")