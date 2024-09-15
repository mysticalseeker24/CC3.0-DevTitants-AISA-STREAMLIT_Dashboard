import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Replace with your actual MongoDB connection string
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client['customer_support_db']
queries_collection = db['customer_queries']

# Streamlit app
st.title('Customer Support Dashboard')

# Recent Queries
st.header('Recent Queries')
recent_queries = list(queries_collection.find().sort('timestamp', -1).limit(10))
if recent_queries:
    df_recent = pd.DataFrame(recent_queries)
    st.table(df_recent[['query', 'status', 'timestamp']])
else:
    st.write("No recent queries found.")

# Query Status
st.header('Query Status')
status_counts = queries_collection.aggregate([
    {'$group': {'_id': '$status', 'count': {'$sum': 1}}}
])
status_df = pd.DataFrame(status_counts)
if not status_df.empty:
    fig = px.pie(status_df, values='count', names='_id', title='Query Status Distribution')
    st.plotly_chart(fig)
else:
    st.write("No query status data available.")

# Query Escalations
st.header('Query Escalations')
escalation_counts = queries_collection.aggregate([
    {'$group': {'_id': '$escalation_level', 'count': {'$sum': 1}}}
])
escalation_df = pd.DataFrame(escalation_counts)
if not escalation_df.empty:
    fig = px.bar(escalation_df, x='_id', y='count', title='Query Escalations')
    st.plotly_chart(fig)
else:
    st.write("No escalation data available.")

# Sentiment Analysis
st.header('Sentiment Analysis')

# Simple sentiment analysis model
vectorizer = CountVectorizer()
classifier = MultinomialNB()

# Fetch queries and responses
queries_and_responses = list(queries_collection.find({}, {'query': 1, 'response': 1, '_id': 0}))
df = pd.DataFrame(queries_and_responses)

if not df.empty:
    # Combine query and response for sentiment analysis
    df['text'] = df['query'] + ' ' + df['response']
    
    # Simple labeling (you might want to use a more sophisticated method)
    df['sentiment'] = df['text'].apply(lambda x: 'positive' if 'thank you' in x.lower() or 'great' in x.lower() else 'negative')
    
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    classifier.fit(X, y)
    
    # Predict sentiment for recent queries
    recent_text = vectorizer.transform(df_recent['query'] + ' ' + df_recent['response'])
    recent_sentiment = classifier.predict(recent_text)
    
    st.write("Sentiment of Recent Queries:")
    st.write(pd.Series(recent_sentiment).value_counts())
    
    # Sentiment trend over time
    df_recent['sentiment'] = recent_sentiment
    df_recent['date'] = pd.to_datetime(df_recent['timestamp']).dt.date
    sentiment_trend = df_recent.groupby('date')['sentiment'].apply(lambda x: (x == 'positive').mean())
    
    fig = px.line(x=sentiment_trend.index, y=sentiment_trend.values, title='Positive Sentiment Trend')
    st.plotly_chart(fig)
else:
    st.write("No data available for sentiment analysis.")

# Historical Resolutions
st.header('Historical Resolutions')
query = st.text_input('Search for a query:')
if query:
    results = queries_collection.find({'query': {'$regex': query, '$options': 'i'}})
    for result in results:
        st.write(f"Query: {result['query']}")
        st.write(f"Response: {result['response']}")
        st.write(f"Status: {result['status']}")
        st.write(f"Timestamp: {result['timestamp']}")
        st.write("---")
