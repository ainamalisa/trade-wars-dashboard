import pandas as pd
import numpy as np

def clean_trade_data(df):
    """Helper function to clean trade data"""
    df = df.dropna(subset=['year', 'value_usd', 'mfn_avg_duty'])
    df = df[df['importer'] != 'ALL Available Markets']
    df['year'] = df['year'].astype(int)
    return df