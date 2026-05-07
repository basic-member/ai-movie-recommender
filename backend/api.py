from fastapi import FastAPI
from database import engine
import models
from routers import admin, recommender, auth, activities
# ---------------------------------------------------------
# 1. APP
# ---------------------------------------------------------

# Initialize database tables
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI Movie Recommender Engine")

app.include_router(auth.router)
app.include_router(activities.router)
app.include_router(admin.router)
app.include_router(recommender.router)

@app.get("/")
def health_check():
    return {"status": "online", "message": "AI Movie Recommender API is running"}