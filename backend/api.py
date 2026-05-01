from fastapi import FastAPI, include_router
from database import engine
import models
from routers import admin, recommender
# ---------------------------------------------------------
# 1. APP
# ---------------------------------------------------------

# Initialize database tables
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI Movie Recommender Engine")

app.include_router(admin.router)
app.include_router(recommender.router)