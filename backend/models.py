from sqlalchemy import Column, Integer, String, ForeignKey, Table, DateTime
from database import Base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

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
    model_id = Column(Integer, nullable=True)

    liked_by_users = relationship("User", secondary=user_likes, back_populates="liked_movies")

    @property
    def like_count(self):
        return len(self.liked_by_users)
    

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    gender = Column(String)
    occu = Column(String)

    liked_movies = relationship("Movie", secondary=user_likes, back_populates="liked_by_users")
    
    search_histories = relationship("SearchHistory", back_populates="user", cascade="all, delete-orphan")

class SearchHistory(Base):
    __tablename__ = "search_histories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    query = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="search_histories")