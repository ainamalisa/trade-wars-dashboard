import streamlit as st
import pandas as pd

def show():
    st.header("Data Overview")
    
    # Data upload and display
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("File uploaded successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
        with col2:
            st.subheader("Data Summary")
            st.write("Shape:", df.shape)
            st.write("Columns:", df.columns.tolist())
        
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        
        st.subheader("Data Types")
        st.write(df.dtypes)