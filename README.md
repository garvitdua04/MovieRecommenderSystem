# Hybrid Movie Recommendation System

A **hybrid movie recommender** built using:

- Content-Based Filtering (TF-IDF + Cosine Similarity)
- Collaborative Filtering (SVD from Surprise library)
- A Hybrid Method (weighted blend of both)

Built with the **MovieLens Latest Small Dataset** (100k ratings) and enhanced with metadata (including posters) from TMDB via `movies_metadata.csv`.

---

### Dataset Used

1. [`ratings.csv`]  
   Contains `userId`, `movieId`, `rating`, `timestamp` (100,000+ ratings)

2. `movies.csv`  
   Maps `movieId` to movie `title` and `genres`

3. `tags.csv`  
   User-generated tags for movies

4. `links.csv`  
   Maps `movieId` to `imdbId` and `tmdbId`

5. `movies_metadata.csv` (from Kaggle)  
   Rich movie metadata including `poster_path`, `overview`, and `genres` for TMDB movie IDs


Links of the datsets:-

(https://grouplens.org/datasets/movielens/) 

(https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

---

###  Features

- **Collaborative Filtering** using SVD from surprise library
- **Content-Based Filtering** using TF-IDF vectorization on genres, titles, and tags  
- **Hybrid Recommender** that combines CF and CBF with a tunable weight parameter  
- **Poster Retrieval** using TMDB poster links  
- **Interactive Streamlit App**

---

### How the Models Work

####  Content-Based Filtering (CBF)

1. Combine `title`, `genres`, and all tags into a single text per movie
2. Use `TfidfVectorizer` to encode movie profiles
3. Use `cosine_similarity` to compute similarity between movies
4. Recommend based on similarity to liked movies

#### Collaborative Filtering (CF)

1. Convert the `ratings.csv` file to Surprise’s `Dataset` format
2. Train an SVD model on user-movie ratings
3. Predict unseen ratings for a given user
4. Recommend top `N` unrated movies with highest predicted ratings

####  Hybrid Recommendation

1. For a given user and liked movies:
   - Get content similarity scores
   - Predict SVD scores for unrated movies
2. Normalize both score vectors
3. Combine them using a weighted average:


### How to Use the App

You can run the Streamlit app locally.

1. Clone the repo
2. Install dependencies
3. Launch the streamlit app (streamlit run app.py)
4. Use the Interface
    - Select liked movies (for CBF/Hybrid)
    - Enter user ID (for CF/Hybrid)
    - Choose recommendation type
    - Set hybrid weight (optional)
    - View recommendations with posters





