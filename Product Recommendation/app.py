import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from PIL import Image
from io import BytesIO
import requests

# ---------- Load and preprocess data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", encoding='ISO-8859-1') #  UK Retail Dataset, https://archive.ics.uci.edu/ml/datasets/Online+Retail
    df.dropna(subset=["Description", "CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    return df

# ---------- Create Item-Item Matrix ----------
def create_item_similarity(df):
    basket = df.pivot_table(index='InvoiceNo', columns='Description', values='Quantity', aggfunc='sum').fillna(0)
    st.title("Item Similarity Matrix")
    print(basket.count())
    similarity = cosine_similarity(basket.T)
    st.title("Similarity Matrix")
    st.dataframe(similarity)
    sim_df = pd.DataFrame(similarity, index=basket.columns, columns=basket.columns)
    st.title("Similarity DataFrame")
    st.dataframe(sim_df)
    return sim_df

# ---------- Recommend Similar Products ----------
def recommend_products(sim_matrix, product, n=5):
    if product not in sim_matrix:
        return []
    recs = sim_matrix[product].sort_values(ascending=False)[1:n+1]
    return recs


# ---------- Streamlit UI ----------

st.set_page_config(page_title="🛒 Product Recommender", layout="wide")
st.title("🛍️ Product Recommendation System")

df = load_data()

st.sidebar.header("🔍 Product Selector")
all_products = sorted(df['Description'].unique())
selected_product = st.sidebar.selectbox("Select a Product", all_products)
print("No of Available products:", df['Description'].unique().shape[0])
# Show recommendations
st.header(f"🔁 Recommendations for: {selected_product}")
sim_matrix = create_item_similarity(df)
recs = recommend_products(sim_matrix, selected_product)

if not recs.empty:
    st.success(f"Top {len(recs)} similar products:")
    for i, (prod, score) in enumerate(recs.items(), 1):
        st.write(f"{i}. **{prod}** — Similarity Score: {round(score, 3)}")
else:
    st.warning("No similar products found.")

# ---------- Aggregation Section ----------
st.header("📊 Product Sales Summary")

top_items = df.groupby("Description")['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
fig = px.bar(top_items, x='Quantity', y='Description', orientation='h', title='Top 10 Bestselling Products')
st.plotly_chart(fig, use_container_width=True)

rev_items = df.copy()
rev_items['Revenue'] = rev_items['Quantity'] * rev_items['UnitPrice']
top_revenue = rev_items.groupby("Description")['Revenue'].sum().sort_values(ascending=False).head(10).reset_index()
fig2 = px.pie(top_revenue, names='Description', values='Revenue', title='Top Revenue-Generating Products')
st.plotly_chart(fig2, use_container_width=True)


