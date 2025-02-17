# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 07:55:56 2025

@author: cvarvin1
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="Online Retail",page_icon="🛒" ,layout="wide")
# Load and clean data (only once using session state)
@st.cache_data
def load_and_clean_data():
    # Load dataset
    df = pd.read_excel("Online Retail.xlsx")

    # Data cleaning
    df = df.dropna(subset=['Description', 'CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9-\s]', '', x))
    df['Totalsales'] = df['Quantity'] * df['UnitPrice']
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce').astype('Int64')
    return df

# Train ML model (only once using session state)
@st.cache_resource
def train_model(data):
    # Prepare data for ML
    predict_data = data.copy()
    predict_data['RepeatPurchase'] = predict_data.duplicated(subset=['CustomerID', 'Description'], keep=False).astype(int)

    X = predict_data[['CustomerID', 'Description']]
    y = predict_data['RepeatPurchase']

    # Factorize description
    X['Description'] = pd.factorize(X['Description'])[0]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    return model

# -------------------- Recommendation System --------------------
@st.cache_resource
def build_recommendation_system(data):
    # Prepare data
    customer_item_matrix = data.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum').fillna(0)

    # Calculate item similarity using cosine similarity
    item_similarity = pd.DataFrame(cosine_similarity(customer_item_matrix.T), 
                                    index=customer_item_matrix.columns, 
                                    columns=customer_item_matrix.columns)
    
    return customer_item_matrix, item_similarity

def recommend_products(item_similarity, purchased_product, n=5):
    """Recommend similar products based on cosine similarity."""
    if purchased_product not in item_similarity.index:
        return "Product not found"
    
    similar_items = item_similarity[purchased_product].sort_values(ascending=False).iloc[1:n+1]
    return similar_items

def recommend_for_customer(customer_id, customer_item_matrix, item_similarity, n=5):
    """Recommend products for a given customer."""
    if customer_id not in customer_item_matrix.index:
        return "Customer not found"
    
    purchased_items = customer_item_matrix.loc[customer_id]
    purchased_items = purchased_items[purchased_items > 0].index
    
    recommendations = {}
    for item in purchased_items:
        recs = recommend_products(item_similarity, item, n)
        if isinstance(recs, pd.Series):
            for rec_item, score in recs.items():
                recommendations[rec_item] = recommendations.get(rec_item, 0) + score
    
    if not recommendations:
        return "No recommendations available."
    
    recommended_products = pd.Series(recommendations).sort_values(ascending=False).head(n)
    max_score = recommended_products.max()
    recommended_products = recommended_products / max_score
    
    return recommended_products


# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = load_and_clean_data()

if 'model' not in st.session_state:
    st.session_state['model'] = train_model(st.session_state['data'])
if 'customer_item_matrix' not in st.session_state or 'item_similarity' not in st.session_state:
    st.session_state['customer_item_matrix'], st.session_state['item_similarity'] = build_recommendation_system(st.session_state['data'])


st.title("🛒 Online Retail Purchases")
# Streamlit page navigation
options = ["📊 EDA Dashboard", "🤖 Purchase Probability", "🎯 Product Recommendation","📊 RFM Analysis"]
page = st.segmented_control("Go to:", options, selection_mode="single", default="📊 EDA Dashboard")

# -------- PAGE 1: EDA Dashboard -------- # 
if page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    df = st.session_state['data']
    
    st.text("📊 Data Overview")
    if st.checkbox("Show Data Sample"):
        st.write(df.head())
        
    st.divider()

    for i in range(3):
        col1, col2 = st.columns(2)

        # Chart 1: Top 10 Products by Total Sales
        if i == 0:
            top_products = df.groupby('Description')['Totalsales'].sum().sort_values(ascending=False).head(10)
            fig1 = px.bar(top_products, x=top_products.index, y=top_products.values, 
                          title="🏆 Top 10 Best-Selling Products", 
                          color=top_products.values, 
                          color_continuous_scale='Viridis')
            fig1.update_layout(xaxis_title="Product", yaxis_title="Total Sales", xaxis_tickangle=-45)
            col1.plotly_chart(fig1)

            # Chart 2: Sales Distribution by Country
            country_sales = df.groupby('Country')['Totalsales'].sum().sort_values(ascending=False).head(10)
            fig2 = px.bar(country_sales, x=country_sales.index, y=country_sales.values, 
                          title="🌍 Top 10 Countries by Sales", 
                          color=country_sales.values, 
                          color_continuous_scale='Cividis')
            fig2.update_layout(xaxis_title="Country", yaxis_title="Total Sales", xaxis_tickangle=-45)
            col2.plotly_chart(fig2)
        
        elif i == 1:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Month'] = df['InvoiceDate'].dt.to_period('M')
            monthly_sales = df.groupby('Month')['Totalsales'].sum().reset_index()
            monthly_sales['Month'] = monthly_sales['Month'].astype(str)
            fig5 = px.line(monthly_sales, x='Month', y='Totalsales', 
                           title="📈 Monthly Sales Trend", 
                           markers=True, 
                           color_discrete_sequence=['#00B894'])
            fig5.update_layout(xaxis_title="Month", yaxis_title="Total Sales")
            col1.plotly_chart(fig5)


            # Chart 4: Sales by Hour of the Day
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Hour'] = df['InvoiceDate'].dt.hour
            
            hourly_sales = df.groupby('Hour')['Totalsales'].sum().reset_index()
            
            fig6 = px.line(hourly_sales, x='Hour', y='Totalsales',
                           title="🕒 Sales by Hour of the Day",
                           markers=True,
                           color_discrete_sequence=['#E17055'])
            fig6.update_layout(xaxis_title="Hour of Day (24H format)", yaxis_title="Total Sales")
            col2.plotly_chart(fig6)

    st.divider()

# -------- PAGE 2: ML Prediction -------- #
if page == "🤖 Purchase Probability":
    st.title("🤖 ML Prediction: Purchase Probability")

    # User input
    customer_id = st.number_input("🔢 Enter Customer ID:", min_value=None, step=None)
    unique_description = df['Description'].unique()
    description = st.multiselect("🛍️ Enter Product Description:",unique_description)

    # Convert description for prediction
    if description:
        description_factor = pd.factorize([description])[0][0]

    # Predict button
    if st.button("🔮 Predict"):
        if customer_id and description:
            # Prepare input for prediction
            input_data = pd.DataFrame({'CustomerID': [customer_id], 'Description': [description_factor]})
            model = st.session_state['model']
            prediction_proba = model.predict_proba(input_data)[0]

            st.write(f"💡 **Probability of Repeat Purchase:**")
            st.write(f"✅ **Repeat:** {prediction_proba[1]*100:.2f}%")
            st.write(f"❌ **No Repeat:** {prediction_proba[0]*100:.2f}%")
        else:
            st.warning("⚠️ Please enter both Customer ID and Product Description.")

    st.divider()

# -------------------- PAGE 3: Product Recommendation --------------------
if page == "🎯 Product Recommendation":
    st.title("🎯 Personalized Product Recommendations")
    customer_id = st.number_input("🔢 Enter Customer ID:", min_value=None, step=None)

    # Recommend button
    if st.button("🎯 Recommend Products"):
        if customer_id:
            # Get recommendations
            recommendations = recommend_for_customer(
                customer_id,
                st.session_state['customer_item_matrix'],
                st.session_state['item_similarity']
            )
            if isinstance(recommendations, pd.Series):
                st.success("🎯 Top Recommended Products:")
                st.write(recommendations)
            else:
                st.warning(f"⚠️ {recommendations}")
        else:
            st.warning("⚠️ Please enter a valid Customer ID.")
            
    st.divider()
    
# -------------------- PAGE 4: RFM Analysis --------------------

if page == "📊 RFM Analysis":
    eda_data = load_and_clean_data()
    
    max_date = eda_data['InvoiceDate'].max()
    rfm = eda_data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (max_date - x.max()).days,'InvoiceNo': 'nunique','Totalsales': 'sum'})
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    
    rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    
    def segment_customers(row):
        if row['RFM_Score'].startswith('4'):
            return 'VIP'
        elif row['RFM_Score'][1] == '4':
            return 'Frequent'
        elif row['RFM_Score'].startswith('1'):
            return 'Lost'
        else:
            return 'Regular'
        
    rfm['Segment'] = rfm.apply(segment_customers, axis=1)
    segment_counts = rfm['Segment'].value_counts().reset_index()
    
    st.dataframe(segment_counts)
    
    for i in range(3):
        col1, col2 = st.columns(2)
    
        # Chart 1: Top 10 Products by Total Sales
        if i == 0:
            fig1 = px.bar(segment_counts, x='index', y='Segment', color='index', 
                 title='Customer Segment Distribution', 
                 labels={'index': 'Segment', 'Segment': 'Count'},
                 color_discrete_sequence=px.colors.qualitative.Pastel)
            col1.plotly_chart(fig1)
            
            fig2 = px.histogram(rfm, x='Recency', nbins=30, title='Recency Distribution', 
                        color_discrete_sequence=['#FF6F61'])
            col2.plotly_chart(fig2)
            
        elif i == 1:
            fig3 = px.histogram(rfm, x='Frequency', nbins=30, title='Frequency Distribution', 
                        color_discrete_sequence=['#6C5CE7'])
            col1.plotly_chart(fig3)
            
            fig4 = px.histogram(rfm, x='Monetary', nbins=30, title='Monetary Distribution', 
                        color_discrete_sequence=['#00B894'])
            col2.plotly_chart(fig4)
            
            st.divider()
    
        













