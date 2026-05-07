from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Movie
from ml_manager import get_movies_df

movies_df = get_movies_df()

router = APIRouter(
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
            # Standard MovieLens/TMDB datasets often have 'id' as the TMDB ID
            t_id = int(row.get('id', 0))
            if t_id == 0 or t_id in seen_tmdb_ids:
                continue
            
            new_movie = Movie(
                id=int(index), 
                tmdb_id=t_id, 
                title=str(row['title'])
            )
            new_movies_objs.append(new_movie)
            seen_tmdb_ids.add(t_id)

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
    return [{"id": m.id, "title": m.title, "tmdb_id": m.tmdb_id} for m in movies]