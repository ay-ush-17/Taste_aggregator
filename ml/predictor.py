import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# --- Configuration ---
DATA_FILE = os.path.join("data", "articles.pkl")
MODEL_FILE = os.path.join("data", "predictor_model.joblib")

# --- Model Training Logic ---

def prepare_data(df):
    """
    Prepares the data for training.
    X = Embeddings
    y = Target (Category)
    """
    print("Preparing data for training...")
    
    # 1. Create the features (X)
    # The features are simply the embedding vectors.
    X = np.vstack(df['embedding'].values)
    
    # 2. Create the target (y)
    # We'll create a simple binary classifier to predict
    # if an article is from 'Ars Technica'.
    # We can get the source from our 'id' column.
    
    # The .contains() check is a bit slow, but robust
    # y = df.index.str.contains('ars_technica').astype(int)
    
    # A faster, more robust way if the ID format is consistent
    y = np.array([1 if 'ars_technica' in idx else 0 for idx in df.index])
    
    if np.sum(y) == 0:
        print("Error: Could not find any 'ars_technica' articles to use as a target.")
        print("Please check the 'id' format in your 'articles.pkl' file.")
        return None, None
        
    print(f"Data prepared: {len(y)} samples, {np.sum(y)} positive (Ars Technica) samples.")
    return X, y

def train_model():
    """
    Loads data, trains a classifier, and saves it.
    """
    try:
        df = pd.read_pickle(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        print("Please run 'python ml/pipeline.py' first.")
        return
        
    X, y = prepare_data(df)
    
    if X is None or y is None:
        return # Error message already printed

    # 1. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 2. Train a simple Logistic Regression model
    # This model is fast and works well on high-dimensional data like embeddings.
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model trained.")

    # 3. Evaluate the model
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation (Test Set) ---")
    print(classification_report(y_test, y_pred, target_names=["Not Ars Technica", "Ars Technica"]))
    
    # 4. Save the trained model to disk
    try:
        joblib.dump(model, MODEL_FILE)
        print(f"Successfully saved trained model to {MODEL_FILE}")
    except Exception as e:
        print(f"Error saving model: {e}")

# --- Test Block ---
if __name__ == "__main__":
    print("Running predictor training script...")
    train_model()