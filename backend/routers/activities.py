from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from oauth2 import get_current_user
import models
import schema
from cache_manager import cache_instance

router = APIRouter(tags=["User Activities"])

@router.post("/movies/{movie_id}/like")
def like_movie(
    movie_id: int, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    """
    Toggle functionality to like or unlike a movie.
    """
    movie = db.query(models.Movie).filter(models.Movie.id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Check if user already liked the movie
    like_query = db.query(models.user_likes).filter(
        models.user_likes.c.user_id == current_user.id, 
        models.user_likes.c.movie_id == movie_id
    )
    
    existing_like = like_query.first()

    if existing_like:
        # Remove like
        like_query.delete(synchronize_session=False)
        db.commit()
        cache_instance.invalidate_user(current_user.id)
        return {"message": "Movie removed from likes"}
    else:
        # Add like
        new_like = models.user_likes.insert().values(
            user_id=current_user.id, 
            movie_id=movie_id
        )
        db.execute(new_like)
        db.commit()
        cache_instance.invalidate_user(current_user.id)
        return {"message": "Movie added to likes"}

@router.patch("/user/update-profile")
def update_profile(
    request: schema.UserUpdate,
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    """
    Update the current user's profile information.
    """
    update_data = request.dict(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(current_user, key, value)

    db.commit()
    db.refresh(current_user)
    cache_instance.invalidate_user(current_user.id)
    return {"message": "Profile updated successfully", "user": current_user}
