from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db, engine
from models import Movie
from schema import User
import models
import pickle
import tensorflow as tf
import numpy as np
import os
import bz2
import traceback

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & AI MODELS LOADING
# ---------------------------------------------------------

# Get the absolute path of the directory where api.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the models folder
# In Docker/Railway, the structure is usually /app/backend/models_data or /app/models_data
MODELS_PATH = os.path.join(BASE_DIR, "models_data")

# Global variables to hold models in memory for faster inference
similar_model = None
movies_df = None
tf_model = None
max_age = 1  # Default value to prevent division by zero
occu_map = {}

print("--- STARTING SYSTEM INITIALIZATION ---")

try:
    # Check if the models directory exists
    if not os.path.exists(MODELS_PATH):
        print(f"❌ FOLDER NOT FOUND: {MODELS_PATH}")
    
    # 1. Load the Compressed Similarity Matrix (bz2 format)
    similarity_file = os.path.join(MODELS_PATH, "similarity.pbz2")
    if os.path.exists(similarity_file):
        file_size = os.path.getsize(similarity_file) / (1024 * 1024)
        print(f"📂 Loading Similarity Matrix: {similarity_file} ({file_size:.2f} MB)")
        with bz2.BZ2File(similarity_file, 'rb') as f:
            similar_model = pickle.load(f)
    else:
        print(f"❌ CRITICAL FILE MISSING: {similarity_file}")

    # 2. Load auxiliary Pickle files (Metadata and Mappings)
    movies_df = pickle.load(open(os.path.join(MODELS_PATH, "movies_list.pkl"), 'rb'))
    max_age = pickle.load(open(os.path.join(MODELS_PATH, "max_age.pkl"), 'rb'))
    occu_map = pickle.load(open(os.path.join(MODELS_PATH, "occupation_map.pkl"), 'rb'))
    
    # 3. Load the Deep Learning Keras Model (TensorFlow)
    print("🤖 Initializing TensorFlow Engine...")
    tf_model = tf.keras.models.load_model(os.path.join(MODELS_PATH, "hybrid_recommender.keras"))
    
    print(f"✅ SYSTEM READY: Successfully loaded {len(movies_df)} movies and all AI components.")

except Exception as e:
    print(f"❌ CRITICAL INITIALIZATION ERROR: {str(e)}")
    print("--- START OF ERROR TRACEBACK ---")
    traceback.print_exc() # Essential for debugging Deployment crashes
    print(f"Searched Path: {MODELS_PATH}")

# ---------------------------------------------------------
# 2. APP & DATABASE SETUP
# ---------------------------------------------------------

# Create database tables on startup if they don't exist
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI Movie Recommender Engine")

@app.get("/setup-db")
def setup_db(db: Session = Depends(get_db)):
    """Synchronizes movie metadata from Pickle files to the SQL database."""
    if movies_df is None:
        raise HTTPException(status_code=500, detail="Dataframe not loaded on server.")

    try:
        # Clear existing entries to prevent primary key conflicts
        db.query(Movie).delete()
        db.commit()

        seen_tmdb_ids = set()
        new_movies_objs = []

        for index, row in movies_df.iterrows():
            tmdb_id_val = int(row['id'])
            if tmdb_id_val in seen_tmdb_ids:
                continue
            
            # Map DataFrame index to Database ID to maintain consistency with similarity matrix
            new_movie = Movie(
                id=int(index), 
                tmdb_id=tmdb_id_val, 
                title=str(row['title'])
            )
            new_movies_objs.append(new_movie)
            seen_tmdb_ids.add(tmdb_id_val)

        # Batch insert for high performance
        db.bulk_save_objects(new_movies_objs)
        db.commit()
        return {"status": "success", "count": len(new_movies_objs)}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database Sync Error: {str(e)}")

@app.get("/setup-db/list")
def get_movies_list(db: Session = Depends(get_db)):
    """Returns a list of all movies for the Frontend dropdown menu."""
    movies = db.query(Movie).all()
    return [{"id": m.id, "title": m.title} for m in movies]

# ---------------------------------------------------------
# 3. HYBRID RECOMMENDATION CORE LOGIC
# ---------------------------------------------------------

@app.post("/recommend/hybrid")
def hybrid_recommendation(movie_id: int, user: User, db: Session = Depends(get_db)):
    """
    Two-Stage Hybrid Engine:
    1. Candidate Generation: Filters top 30 candidates using Content Similarity.
    2. Neural Ranking: Re-ranks candidates using a Deep Learning Keras model.
    """
    try: 
        target_movie = db.query(Movie).filter(Movie.id == movie_id).first()
        if not target_movie:
            raise HTTPException(status_code=404, detail="Movie ID not found in database.")

        # STAGE 1: Get Content-Based Similarities
        distances = similar_model[movie_id]
        candidate_indices = np.argsort(distances)[::-1][1:31] # Exclude the movie itself

        # STAGE 2: User Feature Normalization for Neural Network
        norm_age = user.age / max_age
        gender_code = 1 if user.gender.lower() == "male" else 0
        occ_id = occu_map.get(user.occu, 0)

        recommendation_pool = []
        
        for idx in candidate_indices:
            m = db.query(Movie).filter(Movie.id == int(idx)).first()
            if not m: continue

            # STAGE 3: Re-rank using the Deep Learning Model (if within training range)
            if m.id < 1683 and tf_model:
                prediction = tf_model.predict({
                    "Movie-Input": np.array([m.id]),
                    "Age-Input": np.array([norm_age]),
                    "Gender-Input": np.array([gender_code]),
                    "Occ-Input": np.array([occ_id])
                }, verbose=0)
                final_score = float(prediction[0][0])
            else:
                # Fallback to similarity distance for out-of-training movies
                final_score = float(distances[idx])

            recommendation_pool.append({
                "id": m.id, 
                "title": m.title, 
                "tmdb_id": m.tmdb_id, 
                "score": round(final_score, 2)
            })

        # Sort and return the final Top 5 results
        return sorted(recommendation_pool, key=lambda x: x['score'], reverse=True)[:5]

    except Exception as e:
        traceback.print_exc() # Log detailed error for production monitoring
        raise HTTPException(status_code=500, detail=f"Recommendation Engine Failure: {str(e)}")