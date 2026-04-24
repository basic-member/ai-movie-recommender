from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# آدرس دیتابیس خودت را چک کن (مثلاً sqlite:///./test.db)
SQLALCHEMY_DATABASE_URL = "sqlite:///./recommend.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"❌ خطای دیتابیس: {e}") # این خط بهت می‌گه مشکل چیه
        raise e # این خیلی حیاتیه؛ نباید خطا رو قورت بدی!
    finally:
        db.close()