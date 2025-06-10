import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

def show():
    st.header("Sentiment Analysis on News Articles")
    
    # Download NLTK data
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    @st.cache_data
    def load_data():
        df = pd.read_csv("data/combined_sentiment_data.csv")
        
        # Calculate sentiment if not already present
        if 'sentiment' not in df.columns:
            analyzer = SentimentIntensityAnalyzer()
            df['vader_scores'] = df['text'].apply(lambda x: analyzer.polarity_scores(x))
            df['polarity'] = df['vader_scores'].apply(lambda x: x['compound'])
            df['sentiment'] = df['polarity'].apply(lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral'))
        
        return df
    
    df = load_data()
    
    # Word Cloud
    st.subheader("Word Cloud of Article Titles")
    text = " ".join(df['title'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    usa_colors = ['#3C3B6E', '#ADD8E6', '#B22234']
    
    def usa_color_func(word, *args, **kwargs):
        return usa_colors[len(word) % len(usa_colors)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud.recolor(color_func=usa_color_func), interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                     title="Sentiment Proportion")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                     color=sentiment_counts.index,
                     color_discrete_map={'positive':'#3C3B6E', 'neutral':'#FFFFFF', 'negative':'#B22234'},
                     title="Sentiment Count")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Temporal Trends
    st.subheader("Temporal Trends")
    df['published_date'] = pd.to_datetime(df['published_date'])
    
    tab1, tab2 = st.tabs(["Monthly Trend", "Yearly Trend"])
    
    with tab1:
        monthly_trend = df.groupby(df['published_date'].dt.to_period('M'))['polarity'].mean()
        monthly_trend.index = monthly_trend.index.to_timestamp()
        fig3 = px.line(monthly_trend, title="Monthly Sentiment Trend")
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        yearly_trend = df.groupby(df['published_date'].dt.year)['polarity'].mean()
        fig4 = px.line(yearly_trend, title="Yearly Sentiment Trend")
        st.plotly_chart(fig4, use_container_width=True)

    #fig7
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df['Month'] = df['published_date'].dt.to_period('M')
    monthly_article_counts = df.groupby('Month').size()
    monthly_article_counts.index = monthly_article_counts.index.to_timestamp()

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=monthly_article_counts.index,
        y=monthly_article_counts.values,
        mode='lines+markers',
        name='Articles'
        # line=dict(color='#3C3B6E')
    ))
    fig7.update_layout(
        title="📈 Article Volume Trend by Month",
        xaxis_title="Month",
        yaxis_title="Number of Articles",
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig7, use_container_width=True)

        # Country Analysis
    st.subheader("Country-wise Sentiment")
    agg_df = df.groupby(['country_name', df['published_date'].dt.year])['polarity'].mean().reset_index()
    agg_df.columns = ['country_name', 'year', 'avg_polarity']
    
    fig5 = px.choropleth(agg_df, locations="country_name", locationmode="country names",
                        color="avg_polarity", animation_frame="year",
                        color_continuous_scale=px.colors.diverging.RdYlGn,
                        range_color=[-1, 1], title="Global Sentiment Over Time")
    st.plotly_chart(fig5, use_container_width=True)

    # News Source Analysis - NEW SECTION
    st.subheader("News Source Sentiment Analysis")
    
    # Get the top 20 news sources by article count
    top_sources = df['source'].value_counts().head(20).index
    
    # Calculate average sentiment polarity for the top 20 news sources
    top20_sentiment = (
        df[df['source'].isin(top_sources)]
        .groupby('source')['polarity']
        .mean()
        .sort_values()  # Sort by polarity for better visualization
    )
    
    # Create Plotly figure (better integration with Streamlit than matplotlib)
    fig_source = go.Figure()
    
    # Add bars with conditional coloring
    fig_source.add_trace(go.Bar(
        x=top20_sentiment.index,
        y=top20_sentiment.values,
        marker_color=['#B22234' if val < 0 else '#3C3B6E' for val in top20_sentiment],
        hovertemplate='Source: %{x}<br>Avg Polarity: %{y:.2f}<extra></extra>'
    ))
    
    # Update layout
    fig_source.update_layout(
        title='Average Sentiment Polarity for Top 20 News Sources',
        xaxis_title='News Source',
        yaxis_title='Average Polarity',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=600,
        margin=dict(b=150)  # Add bottom margin for long source names
    )
    
    st.plotly_chart(fig_source, use_container_width=True)