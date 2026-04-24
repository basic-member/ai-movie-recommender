from sqlalchemy import Column, Integer, String
from database import Base

class Movie(Base):
    __tablename__ = "movies"

    # این آی‌دی خود دیتابیس است (Primary Key)
    id = Column(Integer, primary_key=True, index=True)
    
    # آی‌دی مربوط به TMDB (برای گرفتن پوستر و اطلاعات)
    tmdb_id = Column(Integer, unique=True, nullable=False)
    
    # نام فیلم
    title = Column(String, nullable=False)
    
    # آی‌دی مربوط به مدل هوش مصنوعی (همان مترجم ما)
    # اگر فیلمی جدید باشد و در مدل نباشد، این می‌تواند Null باشد
    model_id = Column(Integer, nullable=True)

class User():
    __tablename__ = "users"
    id = Integer
    age = Integer
    gender = str
    occu = str
