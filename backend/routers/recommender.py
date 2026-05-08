from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import Optional
import numpy as np
import traceback

from database import get_db
from models import Movie, user_likes
import models
import schema
from oauth2 import get_current_user_optional 
from ml_manager import get_max_age, get_occu_map, get_similarity_matrix, get_tf_model
from cache_manager import cache_instance

router = APIRouter(prefix="/recommender", tags=["Recommender"])

@router.post("/recommend/hybrid")
def hybrid_recommendation(
    movie_id: int, 
    guest_data: Optional[schema.GuestSession] = Body(None),
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user_optional)
):
    """
    Main Hybrid Recommendation Engine.
    Combines Deep Learning ranking with Collaborative Filtering boosts.
    """
    try: 
        # 0. Cache Check
        user_id = current_user.id if current_user else None
        g_data = guest_data.dict() if guest_data else None
        cache_key = cache_instance.get_key(movie_id, user_id, g_data)
        
        cached_val = cache_instance.get(cache_key)
        if cached_val:
            return cached_val

        # 1. Resolve User Context (Registered User vs Guest)
        if current_user:
            target_age = current_user.age
            target_gender = current_user.gender
            target_occu = current_user.occu
            # Fetch IDs of movies liked by the user
            liked_movies_ids = db.query(user_likes.c.movie_id).filter(
                user_likes.c.user_id == current_user.id
            ).all()
            liked_ids = [m[0] for m in liked_movies_ids]
        elif guest_data:
            target_age = guest_data.age
            target_gender = guest_data.gender
            target_occu = guest_data.occu
            liked_ids = []
        else:
            raise HTTPException(
                status_code=401, 
                detail="Authentication required or guest profile must be provided."
            )

        # 2. Load ML Resources (Lazy Loading)
        similar_model = get_similarity_matrix()
        tf_model = get_tf_model()
        max_age = get_max_age()
        occu_map = get_occu_map()
        
        # 3. Feature Normalization
        norm_age = target_age / max_age
        gender_code = 1 if target_gender.lower() == "male" else 0
        occ_id = occu_map.get(target_occu, 0) 

        # 4. Candidate Generation (Content Similarity)
        distances = similar_model[movie_id]
        candidate_indices = np.argsort(distances)[::-1][1:31]

        # 5. Optimized Batch Query (Avoid 30 separate DB calls)
        movies_query = db.query(Movie).filter(Movie.id.in_(candidate_indices.tolist())).all()
        movie_lookup = {m.id: m for m in movies_query}

        recommendation_pool = []
        
        # 6. Hybrid Re-ranking
        for idx in candidate_indices:
            m = movie_lookup.get(int(idx))
            if not m: continue

            # Part A: Deep Learning Inference (70% weight)
            base_score = 0
            if m.id < 1683 and tf_model:
                prediction = tf_model.predict({
                    "Movie-Input": np.array([m.id]),
                    "Age-Input": np.array([norm_age]),
                    "Gender-Input": np.array([gender_code]),
                    "Occ-Input": np.array([occ_id])
                }, verbose=0)
                base_score = float(prediction[0][0])
            else:
                base_score = float(distances[idx])

            # Part B: Collaborative Filtering Boost (30% weight)
            # If the user has liked movies, boost candidates similar to those liked movies
            like_boost = 0
            if liked_ids:
                sim_with_likes = [similar_model[m.id][l_id] for l_id in liked_ids if l_id < len(similar_model)]
                if sim_with_likes:
                    like_boost = np.mean(sim_with_likes)

            # 7. Weighted Score Fusion
            final_score = (0.875 * base_score) + (0.125 * like_boost)

            recommendation_pool.append({
                "id": m.id, 
                "title": m.title, 
                "tmdb_id": m.tmdb_id,
                "score": round(final_score, 2)
            })

        # Return top 5 sorted by score
        results = sorted(recommendation_pool, key=lambda x: x['score'], reverse=True)[:5]
        cache_instance.set(cache_key, results)
        return results

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Recommender Engine Failure")
@router.get("/occupations/")
def get_occupations_list():
    """Returns the list of occupations for the frontend."""
    occu_map = get_occu_map()
    return sorted(list(occu_map.keys())) if occu_map else ["Other"]
