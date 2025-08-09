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

# ---------- AI Image Generation via Hugging Face ----------
# def generate_product_image(prompt):
#     st.info("Generating image from Hugging Face...")
#     API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
#     headers = {"Authorization": f"Bearer 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"}

#     payload = {"inputs": prompt}
#     response = requests.post(API_URL, headers=headers, json=payload)

#     if response.status_code == 200:
#         image = Image.open(BytesIO(response.content))
#         return image
#     else:
#         st.error(f"❌ Image generation failed: {response.status_code} - {response.text}")
#         return None

# def generate_product_image(prompt):
#     st.info("Generating image from Hugging Face Space...")

#     API_URL = "https://hf.space/embed/multimodalart/latentdiffusion/api/predict/"
#     payload = {
#         "data": [prompt]
#     }

#     try:
#         response = requests.post(API_URL, json=payload)
#         result = response.json()

#         if "data" in result:
#             image_url = result["data"][0]
#             image_response = requests.get(image_url)
#             image = Image.open(BytesIO(image_response.content))
#             return image
#         else:
#             st.error(f"❌ Failed to generate image: {result}")
#             return None

#     except Exception as e:
#         st.error(f"❌ Error generating image: {str(e)}")
#         return None


# Replace YOUR_HUGGINGFACE_TOKEN with your actual token
# HUGGINGFACE_API_TOKEN = "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"

# def generate_product_image(prompt):
#     headers = {
#         "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
#     }

#     API_URL = "https://hf.space/embed/multimodalart/latentdiffusion/api/predict/"
    
#     payload = {
#         "inputs": prompt,
#     }

#     try:
#         response = requests.post(API_URL, headers=headers, json=payload)
#         if response.status_code == 200:
#             image_bytes = response.content
#             img = Image.open(BytesIO(image_bytes))
#             return img
#         else:
#             st.error(f"❌ Image generation failed: {response.status_code} - {response.text}")
#             return None
#     except Exception as e:
#         st.error(f"❌ Error generating image: {str(e)}")
#         return None


# ---------- Streamlit UI ----------

# def main():
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

# # ---------- AI Image Generation ----------
# st.header("🎨 AI Image Generation (via Hugging Face)")
# if st.button("Generate Image for Product"):
#     img = generate_product_image(selected_product)
#     if img:
#         st.image(img, caption=f"AI Generated Image of: {selected_product}", use_container_width=True)

# if __name__ == "__main__":
#     main()
