from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import models
import schema
from database import get_db
from hashing import Hash
from my_token import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
def register(request: schema.UserCreate, db: Session = Depends(get_db)):
    """
    Dedicated registration endpoint.
    """
    # Check if user already exists
    existing_user = db.query(models.User).filter(models.User.email == request.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")
    
    new_user = models.User(
        email=request.email,
        name=request.name,
        password=Hash.bcrypt(request.password),
        age=request.age,
        gender=request.gender,
        occu=request.occu
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Auto-login after registration
    access_token = create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer", "email": new_user.email}

@router.post("/login")
def login(request: schema.UserLogin, db: Session = Depends(get_db)):
    """
    Dedicated login endpoint (JSON).
    """
    user = db.query(models.User).filter(models.User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found. Please register.")
    
    if not Hash.verify(request.password, user.password):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer", "email": user.email}
