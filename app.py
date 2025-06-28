import streamlit as st
import pandas as pd
from movie_recommendation import (
    cb_movies_df, movies, ratings, merged_df,
    get_cbf_recommendations, get_cf_recommendations, hybrid_recommendations
)

# Configure Streamlit page
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("Hybrid Movie Recommendation System")

# --- User Input Section ---

# Select movies for content-based and hybrid modes
liked_movies = st.multiselect(
    "Select Movies You Like:",
    options=sorted(cb_movies_df['title'].tolist())
)

# Input User ID for collaborative and hybrid modes
user_id = st.number_input("Enter Your User ID:", min_value=1, step=1)

# Choose recommendation mode
mode = st.radio(
    "Select Recommendation Type:",
    options=["Content-Based", "Collaborative Filtering", "Hybrid"],
    horizontal=True
)

# Optional hybrid weighting (only appears if Hybrid is selected)
alpha = 0.5
if mode == "Hybrid":
    alpha = st.slider("Hybrid Weight (Content-Based %):", 0.0, 1.0, 0.5, step=0.05)

# Number of recommendations to display
top_n = st.slider("Number of Recommendations:", 5, 20, 10,step=1)

# Ensure consistency of movieId column
merged_df['movieId'] = merged_df['movieId'].astype(int)

# --- Recommendation Logic ---
recommendations = None

if st.button("Get Recommendations"):
    if mode == "Content-Based":
        if liked_movies:
            recommendations = get_cbf_recommendations(liked_movies, top_n=top_n)
        else:
            recommendations = "Please select at least one movie."

    elif mode == "Collaborative Filtering":
        if user_id:
            recommendations = get_cf_recommendations(user_id, top_n=top_n)
        else:
            recommendations = "Please enter a valid User ID."

    elif mode == "Hybrid":
        if user_id and liked_movies:
            recommendations = hybrid_recommendations(user_id, liked_movies, top_n=top_n, alpha=alpha)
        else:
            recommendations = "Please enter a valid User ID and select liked movies."

    # --- Result Display ---
    if isinstance(recommendations, str):
        st.warning(recommendations)

    elif isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        for _, row in recommendations.iterrows():
            movie_id = row['movieId']
            title = row['title']

            # Try to fetch poster URL
            poster_row = merged_df[merged_df['movieId'] == movie_id]

            if not poster_row.empty and pd.notnull(poster_row.iloc[0]['poster_url']):
                st.image(poster_row.iloc[0]['poster_url'], width=150, caption=title)
            else:
                st.markdown(f"**{title}** – Poster not available.")
            st.markdown("---")
    else:
        st.warning("No recommendations available or unexpected format.")
