import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Configuration ---
# The path to the article database we created in pipeline.py
DATA_FILE = os.path.join("data", "articles.pkl")

# --- Recommendation Logic ---

def load_data():
    """
    Loads the article DataFrame from the pickle file.
    """
    try:
        df = pd.read_pickle(DATA_FILE)
        # We need to make sure the 'embedding' column is stacked as a NumPy matrix
        # for efficient cosine similarity calculation.
        embeddings_matrix = np.vstack(df['embedding'].values)
        return df, embeddings_matrix
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        print("Please run 'python ml/pipeline.py' first to create the article database.")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def get_recommendations(liked_article_ids, df, embeddings_matrix, top_n=5):
    """
    Recommends articles based on a list of liked article IDs.
    
    Args:
        liked_article_ids (list): A list of article IDs (strings) that the user likes.
        df (pd.DataFrame): The main article DataFrame.
        embeddings_matrix (np.array): The pre-computed matrix of all article embeddings.
        top_n (int): The number of recommendations to return.
    """
    try:
        # 1. Get the embedding vectors for the liked articles
        liked_indices = [df.index.get_loc(id) for id in liked_article_ids if id in df.index]
        if not liked_indices:
            print("Warning: None of the liked article IDs were found in the database.")
            return []
            
        liked_vectors = embeddings_matrix[liked_indices]
        
        # 2. Create a single "user preference vector" by averaging the liked vectors
        user_preference_vector = np.mean(liked_vectors, axis=0).reshape(1, -1)
        
        # 3. Calculate cosine similarity between the user's vector and all articles
        # This is the core of the recommender!
        similarities = cosine_similarity(user_preference_vector, embeddings_matrix)
        
        # 4. Get the indices of the top_n most similar articles
        # We use .argsort() to get sorted indices, then [-top_n-1:-1] to get the top ones
        # We exclude the most similar one if it's one of the articles they already liked
        
        # Get the indices of the top (top_n + len(liked_article_ids)) most similar articles
        # This ensures we have enough to filter from
        n_results = top_n + len(liked_article_ids)
        top_indices = similarities[0].argsort()[-n_results:][::-1] # [::-1] reverses to descending order

        # 5. Filter out articles the user has already liked
        recommendations = []
        for idx in top_indices:
            if df.index[idx] not in liked_article_ids:
                recommendations.append(df.index[idx])
            
            if len(recommendations) >= top_n:
                break
                
        # Return a list of recommended article IDs and their titles
        rec_data = df.loc[recommendations][['title', 'url']]
        return rec_data
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []

# --- Test Block ---
# This code will only run when you execute this script directly:
# python ml/recommender.py
if __name__ == "__main__":
    print("Testing recommender system...")
    
    # 1. Load the data
    articles_df, embeddings = load_data()
    
    if articles_df is not None:
        print(f"Loaded {len(articles_df)} articles.")
        
        # 2. Create a dummy list of "liked" articles
        # We'll just pick the first two articles from our database
        liked_ids = articles_df.index.to_list()[:2]
        print(f"\nRecommending based on {len(liked_ids)} liked articles:")
        print(f"  - {articles_df.loc[liked_ids[0]]['title']}")
        print(f"  - {articles_df.loc[liked_ids[1]]['title']}")

        # 3. Get recommendations
        recommendations = get_recommendations(liked_ids, articles_df, embeddings, top_n=5)
        
        if recommendations is not None:
            print(f"\nTop 5 Recommendations:")
            print(recommendations)
