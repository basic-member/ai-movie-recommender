from sqlalchemy import Column, Integer, String, ForeignKey, Table
from database import Base
from sqlalchemy.orm import relationship

# Association table for User-Movie Likes (Many-to-Many)
user_likes = Table(
    "user_likes",
    Base.metadata,
    Column("user_id", ForeignKey("users.id"), primary_key=True),
    Column("movie_id", ForeignKey("movies.id"), primary_key=True),
)

class Movie(Base):
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True, index=True)
    tmdb_id = Column(Integer, unique=True, nullable=False)
    title = Column(String, nullable=False)
    model_id = Column(Integer, nullable=True) # ID mapping for the ML model

    liked_by_users = relationship("User", secondary=user_likes, back_populates="liked_movies")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    occu = Column(String)

    liked_movies = relationship("Movie", secondary=user_likes, back_populates="liked_by_users")