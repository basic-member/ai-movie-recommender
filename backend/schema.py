from pydantic import BaseModel, EmailStr, constr, Field
from typing import List, Optional

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: constr(min_length=8, max_length=72)
    age: int = Field(..., ge=5, le=100, description="Age of the user")
    gender: str
    occu: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=5, le=100)
    gender: Optional[str] = None
    occu: Optional[str] = None

class MovieResponse(BaseModel):
    id: int
    title: str
    tmdb_id: Optional[int] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class User(BaseModel):
    id: int
    email: EmailStr
    name: str
    age: int = Field(..., ge=5, le=100)
    gender: str
    occu: str

    class Config:
        from_attributes = True

class GuestSession(BaseModel):
    age: int = Field(..., ge=5, le=100)
    gender: str
    occu: str