import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import dump

# Load dataset from a ZIP file
@st.cache_data
def load_data():
    # Define the ZIP file and CSV file names
    zip_file = "amazon.csv.zip"
    csv_file = "amazon.csv"

    # Extract the ZIP file if not already extracted
    if not os.path.exists(csv_file):
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extract(csv_file)

    # Load the CSV file into a DataFrame
    return pd.read_csv(csv_file)

# Load the dataset
df = load_data()

# Load the SVD model
_, model = dump.load('svd_model.pkl')

# Create a TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['about_product'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Content-Based Recommendation Function
def content_based_recommendations(product_name, num_recommendations=5):
    try:
        idx = df[df['product_name'].str.contains(product_name, case=False, na=False)].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        return df.iloc[sim_indices][['product_name', 'rating', 'discounted_price']]
    except IndexError:
        return "Product not found. Please enter a valid product name."

# Collaborative Filtering Recommendation Function
def collaborative_recommendations(user_id, num_recommendations=5):
    product_ids = df['product_id'].unique()
    predictions = [(pid, model.predict(user_id, pid).est) for pid in product_ids]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:num_recommendations]
    return df[df['product_id'].isin([x[0] for x in top_predictions])][['product_name', 'rating', 'discounted_price']]

# Hybrid Recommendation Function
def hybrid_recommendations(user_id, product_name, num_recommendations=5, content_weight=0.5, collab_weight=0.5):
    try:
        product_ids = df['product_id'].unique()
        collab_predictions = [(pid, model.predict(user_id, pid).est) for pid in product_ids]
        collab_predictions = sorted(collab_predictions, key=lambda x: x[1], reverse=True)
        top_collab = collab_predictions[:num_recommendations]
        collab_recs = df[df['product_id'].isin([x[0] for x in top_collab])]

        idx = df[df['product_name'].str.contains(product_name, case=False, na=False)].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        content_recs = df.iloc[sim_indices]

        content_recs['score'] = content_weight * content_recs.index.map(lambda i: sim_scores[i][1])
        collab_recs['score'] = collab_weight * collab_recs['product_id'].map(
            lambda pid: dict(collab_predictions)[pid] if pid in dict(collab_predictions) else 0
        )

        final_recs = pd.concat([content_recs, collab_recs])
        final_recs = final_recs.groupby('product_id').max().reset_index()
        final_recs = final_recs.sort_values(by='score', ascending=False).head(num_recommendations)

        return final_recs[['product_name', 'rating', 'discounted_price', 'score']]
    except IndexError:
        return "Product not found or invalid user ID. Please provide valid inputs."

# Streamlit App
st.title("Hybrid Recommendation System")
st.write("Get recommendations based on product or user preferences!")

option = st.selectbox("Choose a recommendation type:", ["Content-Based", "Collaborative", "Hybrid"])

if option == "Content-Based":
    product_name = st.text_input("Enter Product Name:")
    if st.button("Get Recommendations"):
        if product_name:
            recommendations = content_based_recommendations(product_name)
            st.write(recommendations)
        else:
            st.error("Please enter a valid product name!")

elif option == "Collaborative":
    user_id = st.text_input("Enter User ID:")
    if st.button("Get Recommendations"):
        if user_id:
            recommendations = collaborative_recommendations(user_id)
            st.write(recommendations)
        else:
            st.error("Please enter a valid user ID!")

else:
    user_id = st.text_input("Enter User ID:")
    product_name = st.text_input("Enter Product Name:")
    if st.button("Get Recommendations"):
        if user_id and product_name:
            recommendations = hybrid_recommendations(user_id, product_name)
            st.write(recommendations)
        else:
            st.error("Please enter both User ID and Product Name!")
