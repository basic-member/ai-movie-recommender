from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Movie
from schema import User  
import traceback
import numpy as np
from ml_manager import get_max_age, get_occu_map, get_similarity_matrix, get_tf_model

router = APIRouter(prefix="/recommender", tags=["Recommender"])

# ---------------------------------------------------------
# 3. HYBRID RECOMMENDATION CORE LOGIC
# ---------------------------------------------------------

@router.post("/recommend/hybrid")
def hybrid_recommendation(movie_id: int, user: User, db: Session = Depends(get_db)):
    """
    Hybrid Recommendation Engine:
    Optimized with Lazy Loading to ensure memory efficiency under high load.
    """
    try: 
        # 1. Access models ONLY when needed (Lazy Loading)
        # This ensures RAM is only used when an actual request hits this endpoint.
        similar_model = get_similarity_matrix()
        tf_model = get_tf_model()
        max_age = get_max_age()
        occu_map = get_occu_map()

        target_movie = db.query(Movie).filter(Movie.id == movie_id).first()
        if not target_movie:
            raise HTTPException(status_code=404, detail="Movie not found.")

        # STAGE 1: Candidate Generation (Content Similarity)
        # Fast vector lookup using the pre-computed matrix
        distances = similar_model[movie_id]
        candidate_indices = np.argsort(distances)[::-1][1:31]

        # STAGE 2: Profile Normalization
        norm_age = user.age / max_age
        gender_code = 1 if user.gender.lower() == "male" else 0
        occ_id = occu_map.get(user.occu, 0)

        recommendation_pool = []
        
        # STAGE 3: Neural Re-ranking
        for idx in candidate_indices:
            m = db.query(Movie).filter(Movie.id == int(idx)).first()
            if not m: continue

            # Deep Learning inference for candidates within the trained distribution
            if m.id < 1683 and tf_model:
                prediction = tf_model.predict({
                    "Movie-Input": np.array([m.id]),
                    "Age-Input": np.array([norm_age]),
                    "Gender-Input": np.array([gender_code]),
                    "Occ-Input": np.array([occ_id])
                }, verbose=0)
                final_score = float(prediction[0][0])
            else:
                # Fallback to similarity score for new/out-of-range movies
                final_score = float(distances[idx])

            recommendation_pool.append({
                "id": m.id, 
                "title": m.title, 
                "tmdb_id": m.tmdb_id, 
                "score": round(final_score, 2)
            })

        # Return Top 5 candidates sorted by predicted neural score
        return sorted(recommendation_pool, key=lambda x: x['score'], reverse=True)[:5]

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")