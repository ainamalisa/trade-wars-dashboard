# streamlit_app.py
import streamlit as st

# Streamlit page config - MUST be first
st.set_page_config(page_title="TradeWars Dashboard", layout="wide")

# Now import sections after page config
from sections import overview, data, trends, sentiment, correlation, conclusion, predict

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

sections = [
    ("📌 Project Overview", overview),
    ("📁 Data Overview", data),
    ("📈 Trends in Trade Volume", trends),
    ("💬 Sentiment Analysis", sentiment),
    ("📊 Correlation Analysis", correlation),
    ("📉 Predictive Modeling", predict),
    ("✅ Conclusion & Recommendations", conclusion)
]

# --- Initialize session state ---
if "active_section" not in st.session_state:
    st.session_state.active_section = overview

# --- Sidebar layout ---
st.sidebar.title("📊 TradeWars Dashboard")


for label, section_module in sections:
    if st.sidebar.button(label):
        st.session_state.active_section = section_module

st.session_state.active_section.show()