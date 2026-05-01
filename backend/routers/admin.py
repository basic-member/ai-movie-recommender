from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Movie
from ml_manager import get_movies_df

movies_df = get_movies_df()

router = APIRouter(
    prefix="/data",
    tags=["data"]
)

@router.get("/setup-db")
def setup_db(db: Session = Depends(get_db)):
    """Synchronizes movie metadata from the pickle file to the SQL database."""
    if movies_df is None:
        raise HTTPException(status_code=500, detail="Dataframe not loaded.")

    try:
        # Clear existing movies to avoid duplicate key errors
        db.query(Movie).delete()
        db.commit()

        seen_tmdb_ids = set()
        new_movies_objs = []

        for index, row in movies_df.iterrows():
            tmdb_id_val = int(row['id'])
            if tmdb_id_val in seen_tmdb_ids:
                continue
            
            # Map index to ID to maintain alignment with similarity matrix
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

@router.get("/setup-db/list")
def get_movies_list(db: Session = Depends(get_db)):
    """Fetches all movies for UI selection."""
    movies = db.query(Movie).all()
    return [{"id": m.id, "title": m.title} for m in movies]