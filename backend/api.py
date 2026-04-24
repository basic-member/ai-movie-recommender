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

# ---------------------------------------------------------
# 1. PATH CONFIGURATION & MODELS LOADING
# ---------------------------------------------------------
# This ensures files are found regardless of where you run uvicorn from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If api.py is in 'backend' folder and data is in 'models_data' at root:
MODELS_PATH = os.path.join(os.path.dirname(BASE_DIR), "models_data")

# Initialize global variables to None
similar_model = None
movies_df = None
tf_model = None
max_age = 1 # Avoid division by zero
occu_map = {}

try:
    # Loading Pickles
    similar_model = pickle.load(open(os.path.join(MODELS_PATH, "similarity.pkl"), 'rb'))
    movies_df = pickle.load(open(os.path.join(MODELS_PATH, "movies_list.pkl"), 'rb'))
    max_age = pickle.load(open(os.path.join(MODELS_PATH, "max_age.pkl"), 'rb'))
    occu_map = pickle.load(open(os.path.join(MODELS_PATH, "occupation_map.pkl"), 'rb'))
    
    # Loading Keras Model
    tf_model = tf.keras.models.load_model(os.path.join(MODELS_PATH, "hybrid_recommender.keras"))
    
    print(f"✅ SYSTEM READY: Loaded {len(movies_df)} movies and all AI models.")
except Exception as e:
    print(f"❌ CRITICAL INITIALIZATION ERROR: {e}")
    print(f"Looking for files in: {MODELS_PATH}")

# ---------------------------------------------------------
# 2. APP & DATABASE SETUP
# ---------------------------------------------------------
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI Movie Recommender API")

@app.get("/setup-db")
def setup_db(db: Session = Depends(get_db)):
    """Synchronizes the Pickle data with the SQLite Database."""
    if movies_df is None:
        raise HTTPException(status_code=500, detail="DataFrame not loaded. Check backend logs.")

    try:
        # Clear database to prevent ID conflicts
        db.query(Movie).delete()
        db.commit()

        seen_tmdb_ids = set()
        new_movies_objs = []

        for index, row in movies_df.iterrows():
            tmdb_id_val = int(row['id'])
            if tmdb_id_val in seen_tmdb_ids:
                continue
            
            # Use dataframe index as ID to sync with similarity matrix
            new_movie = Movie(
                id=int(index), 
                tmdb_id=tmdb_id_val, 
                title=str(row['title'])
            )
            new_movies_objs.append(new_movie)
            seen_tmdb_ids.add(tmdb_id_val)

        db.bulk_save_objects(new_movies_objs)
        db.commit()
        return {"status": "success", "count": len(new_movies_objs)}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/setup-db/list")
def get_movies_list(db: Session = Depends(get_db)):
    """Returns movie list for the Frontend dropdown."""
    movies = db.query(Movie).all()
    return [{"id": m.id, "title": m.title} for m in movies]

# ---------------------------------------------------------
# 3. RECOMMENDATION ENGINE
# ---------------------------------------------------------
@app.post("/recommend/hybrid")
def hybrid_recommendation(movie_id: int, user: User, db: Session = Depends(get_db)):
    """The core engine combining Similarity Matrix and Deep Learning."""
    try: 
        # Check if movie exists
        target_movie = db.query(Movie).filter(Movie.id == movie_id).first()
        if not target_movie:
            raise HTTPException(status_code=404, detail="Movie ID not found in DB.")

        # Step 1: Content Similarity
        distances = similar_model[movie_id]
        candidate_indices = np.argsort(distances)[::-1][1:31]

        # Step 2: Prepare User Features
        norm_age = user.age / max_age
        gender_code = 1 if user.gender.lower() == "male" else 0
        occ_id = occu_map.get(user.occu, 0)

        recommendation_pool = []
        
        for idx in candidate_indices:
            m = db.query(Movie).filter(Movie.id == int(idx)).first()
            if not m: continue

            # Step 3: Neural Network Ranking (if in training range)
            if m.id < 1683 and tf_model:
                prediction = tf_model.predict({
                    "Movie-Input": np.array([m.id]),
                    "Age-Input": np.array([norm_age]),
                    "Gender-Input": np.array([gender_code]),
                    "Occ-Input": np.array([occ_id])
                }, verbose=0)
                final_score = float(prediction[0][0])
            else:
                # Fallback for newer movies
                final_score = float(distances[idx])

            recommendation_pool.append({
                "id": m.id, "title": m.title, "tmdb_id": m.tmdb_id, "score": round(final_score, 2)
            })

        # Return top 5 sorted by score
        return sorted(recommendation_pool, key=lambda x: x['score'], reverse=True)[:5]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")