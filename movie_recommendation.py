"""
 Hybrid Movie Recommendation System

This project builds a hybrid recommendation engine using:

- Collaborative Filtering (SVD)
- Content-Based Filtering (TF-IDF + Cosine Similarity)
- A hybrid method combining both

I have used the **MovieLens Latest Small Dataset** (100k ratings) from [GroupLens] and movies_metadata.csv
from kaggle as it contains all info about the movies  and it also has poster links of the movies.
"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy

# Load files
ratings = pd.read_csv("data/ratings.csv")     # userId, movieId, rating, timestamp
movies = pd.read_csv("data/movies.csv")       # movieId, title, genres
tags = pd.read_csv("data/tags.csv")           # userId, movieId, tag, timestamp
links = pd.read_csv("data/links.csv")         # movieId, imdbId, tmdbId
metadata = pd.read_csv("data/movies_metadata.csv", low_memory=False) #all info


# Drop rows with missing or invalid imdb_id
metadata = metadata[metadata['imdb_id'].notnull()]
metadata = metadata[metadata['imdb_id'].str.startswith('tt')]

# Convert 'tt1234567' → 1234567 (int)
metadata['imdb_id_num'] = metadata['imdb_id'].str[2:].astype(int)


# Merge links with metadata using imdbId and imdb_id_num
merged_df = pd.merge(links, metadata, left_on='imdbId', right_on='imdb_id_num', how='left')

base_url = "https://image.tmdb.org/t/p/w500"

# keep only those movies that have a valid poster path
merged_df = merged_df[merged_df['poster_path'].notnull()]

# Create a new column 'poster_url'
merged_df['poster_url'] = base_url + merged_df['poster_path'].astype(str)

# Genres are originally pipe-separated like "Action|Adventure"
# We'll convert to space-separated: "Action Adventure" for text processing

movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)



"""Build Content-Based Filtering DataFrame

Created a final clean dataframe `cb_movies_df` that contains:
- `movieId`
- `title`
- `cb_text`: concatenation of `title`, `genres`, and **all user tags** combined per movie

This ensures each movie has a rich content profile for TF-IDF modeling.

"""

# Group all tags for each movie into a single string
tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Merge movies with grouped tags using LEFT JOIN
cb_movies_df = pd.merge(movies, tags_grouped, on='movieId', how='left')

# Fill missing tag column with empty string (for movies with no tags)
cb_movies_df['tag'] = cb_movies_df['tag'].fillna('')

# Create combined content text (cb_text)
cb_movies_df['cb_text'] = (
    cb_movies_df['title'].fillna('') + ' ' +
    cb_movies_df['genres'].fillna('') + ' ' +
    cb_movies_df['tag'].fillna('')
)

# Keep only needed columns for CBF
cb_movies_df = cb_movies_df[['movieId', 'title', 'cb_text']]



"""TF-IDF Vectorization + Cosine Similarity

Used TF-IDF to convert movie content (`cb_text`) into numeric vectors, and cosine similarity to compare them. This allows us to recommend similar movies based purely on metadata and tags.

"""
# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the cb_text
tfidf_matrix = tfidf.fit_transform(cb_movies_df['cb_text'])

# Compute pairwise cosine similarity between all movies
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

""" Build Recommendation Function

Created a function that:
1. Takes a movie title as input
2. Finds its index in the dataframe
3. Looks up cosine similarity scores for that movie
4. Returns top N most similar movies (excluding the input itself)

"""

#  Create a reverse map of movie title to index
title_to_index = pd.Series(cb_movies_df.index, index=cb_movies_df['title'])

# Recommendation function
def get_similar_movies(title, top_n=10):
    # Check if title exists
    if title not in title_to_index:
        return f"Movie '{title}' not found in dataset."

    # Get the index of the movie
    idx = title_to_index[title]

    # Get similarity scores for this movie with all others
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort movies by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first one (it's the same movie)
    sim_scores = sim_scores[1:top_n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the movie titles
    return cb_movies_df[['title']].iloc[movie_indices].reset_index(drop=True)


"""##  Multi-Movie Content-Based Recommender

This version:
- Takes a list of movies the user likes
- Retrieves each movie's content similarity vector
- Averages them into a single "user interest profile"
- Returns top N recommended movies based on this combined profile

"""

def get_cbf_recommendations(liked_titles, top_n=10):

    # Filter out movies not found
    valid_titles = [title for title in liked_titles if title in title_to_index]

    if not valid_titles:
        return "None of the selected movies were found in the dataset."

    # Get indices of liked movies
    liked_indices = [title_to_index[title] for title in valid_titles]

    # Average the cosine similarity vectors of liked movies
    user_profile = cosine_sim_matrix[liked_indices].mean(axis=0)

    # Get list of scores with their index
    sim_scores = list(enumerate(user_profile))

    # Remove the movies the user already liked
    liked_set = set(liked_indices)
    sim_scores = [score for score in sim_scores if score[0] not in liked_set]

    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N recommended movie indices
    top_indices = [idx for idx, _ in sim_scores[:top_n]]

    # Get movieId and title from cb_movies_df using indices
    recommended_movies = cb_movies_df.loc[top_indices, ['movieId', 'title']].reset_index(drop=True)

    return recommended_movies



""" Collaborative Filtering (SVD)

 Built a recommendation engine based on user rating patterns Singular Value Decomposition (SVD) via the Surprise library.

Instead of relying on content, this approach learns:
- What types of movies a user prefers (latent user features)
- What hidden characteristics a movie has (latent item features)

The key idea is:
"If User A and B both liked Movie X, and User A also liked Y, then maybe B will like Y too."

"""


""" Create User-Item Matrix

Reshaped the ratings data into a matrix format:

- Rows: Unique users
- Columns: Unique movies
- Values: Rating (0.5 to 5.0 stars)

 ill missing values with 0 temporarily.

"""

# Pivot ratings to create user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Fill missing values with 0 (assumes unrated movies)
user_item_matrix_filled = user_item_matrix.fillna(0)

# Define the reader with rating scale 0.5 to 5.0
reader = Reader(rating_scale=(0.5, 5.0))

# Load dataset into Surprise format
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Create and train SVD model
svd_model = SVD(random_state=42)
svd_model.fit(trainset)

# Predict on test set
predictions = svd_model.test(testset)


# Build recommendation function
def get_cf_recommendations(user_id, top_n=10):
  
    if user_id not in ratings['userId'].unique():
        return f"User ID {user_id} not found in the dataset."

    # Get all movie IDs
    all_movie_ids = ratings['movieId'].unique()

    # Get movies already rated by the user
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].values

    # Predict ratings for unrated movies
    predictions = [
        svd_model.predict(user_id, movie_id)
        for movie_id in all_movie_ids if movie_id not in rated_movie_ids
    ]

    # Sort by predicted rating (descending)
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]

    # Prepare DataFrame
    results = []
    for pred in top_predictions:
        movie_id = int(pred.iid)
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        results.append({
            'movieId': movie_id,
            'title': title,
            'predicted_rating': round(pred.est, 2)
        })

    return pd.DataFrame(results)




# Hybrid recommendation function

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def hybrid_recommendations(user_id, liked_titles, top_n=10, alpha=0.5):

    if user_id not in ratings['userId'].unique():
        return f"User ID {user_id} not found in the dataset."

    # Filter liked movies that exist in the model
    valid_titles = [title for title in liked_titles if title in title_to_index]
    if not valid_titles:
        return pd.DataFrame(columns=["movieId", "title", "cb_score", "cf_score", "hybrid_score"])

    # Compute average content-based profile
    liked_indices = [title_to_index[title] for title in valid_titles]
    user_profile = cosine_sim_matrix[liked_indices].mean(axis=0)

    # Create DataFrame of content-based scores (excluding liked movies)
    liked_set = set(liked_indices)
    sim_scores = [(idx, score) for idx, score in enumerate(user_profile) if idx not in liked_set]

    cb_df = pd.DataFrame(sim_scores, columns=['index', 'cb_score'])
    cb_df['movieId'] = movies.loc[cb_df['index'], 'movieId'].values
    cb_df = cb_df[['movieId', 'cb_score']]

    # Collaborative Filtering: Predict scores for unrated movies
    all_movie_ids = ratings['movieId'].unique()
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].values
    unrated_ids = set(all_movie_ids) - set(rated_movie_ids)

    cf_scores = []
    for movie_id in unrated_ids:
        pred = svd_model.predict(user_id, movie_id)
        cf_scores.append((movie_id, pred.est))

    cf_df = pd.DataFrame(cf_scores, columns=['movieId', 'cf_score'])

    # Merge both scores
    hybrid_df = pd.merge(cb_df, cf_df, on='movieId')
    if hybrid_df.empty:
        return pd.DataFrame(columns=["movieId", "title", "cb_score", "cf_score", "hybrid_score"])

    # Normalize both score columns to [0, 1]
    scaler = MinMaxScaler()
    hybrid_df[['cb_score', 'cf_score']] = scaler.fit_transform(hybrid_df[['cb_score', 'cf_score']])

    # Compute weighted hybrid score
    hybrid_df['hybrid_score'] = alpha * hybrid_df['cb_score'] + (1 - alpha) * hybrid_df['cf_score']

    # Add movie titles
    hybrid_df = pd.merge(hybrid_df, movies[['movieId', 'title']], on='movieId')
    hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False).head(top_n)
    hybrid_df.reset_index(drop=True, inplace=True)

    return hybrid_df[['movieId', 'title', 'cb_score', 'cf_score', 'hybrid_score']]




