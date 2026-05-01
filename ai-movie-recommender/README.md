# 🎬 AI-Powered Hybrid Movie Recommender

An enterprise-grade movie recommendation engine built with **FastAPI** and **TensorFlow**. This system features a sophisticated two-stage pipeline (Retrieval & Ranking) and is fully containerized for high-performance deployment.

### 📺 Project Preview
[![Hybrid Recommender Demo](https://img.shields.io/badge/🎥-View_Live_Demo-red?style=for-the-badge)](https://github.com/basic-member/ai-movie-recommender)

## 🚀 Key Engineering Highlights

*   **Modular API Design:** Decoupled backend logic using FastAPI `APIRouter` for clean and maintainable code.
*   **Lazy Loading Engine:** Custom `ml_manager.py` implementation ensures heavy TensorFlow models are loaded into memory only upon the first request, optimizing RAM usage and startup speed.
*   **Hybrid Intelligence:** Integrates **Content-Based Filtering** (Cosine Similarity) with **Neural Collaborative Filtering** (Deep Learning) for personalized re-ranking.
*   **Full Dockerization:** Multi-container orchestration using `docker-compose` for seamless environment parity between development and production.
*   **Production-Ready DB:** Managed data persistence using SQLAlchemy ORM with SQLite, including bulk-sync capabilities.

## 🧠 Technical Architecture

The system utilizes a professional **Two-Stage Pipeline** to provide recommendations:
1.  **Stage 1: Candidate Generation (Retrieval):** Uses a pre-computed similarity matrix (`similarity.pbz2`) to identify the top 30 candidates from the database.
2.  **Stage 2: Neural Re-ranking (Scoring):** These candidates are fed into a TensorFlow model (`hybrid_recommender.keras`) along with user demographics (Age, Gender, Occupation) to predict the final Top-5 results.

## 🐳 Docker Deployment

To launch the entire ecosystem (FastAPI Backend + Streamlit Frontend) with a single command:

1.  **Clone the Repository:**
    
```bash
    git clone [https://github.com/basic-member/ai-movie-recommender.git](https://github.com/basic-member/ai-movie-recommender.git)
    cd ai-movie-recommender
    ```

2.  **Setup Environment:**
    Ensure your model files are in `backend/models_data/`.

3.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    *   **Backend API:** `http://localhost:8000`
    *   **Frontend UI:** `http://localhost:8501`

## 📂 Project Structure
```text
.
├── backend/
│   ├── models_data/       # AI Models & Data Assets (.keras, .pkl, .pbz2)
│   ├── routers/           # Modular API endpoints (admin.py, recommender.py)
│   ├── database.py        # SQLAlchemy configuration
│   ├── ml_manager.py      # Lazy loading & Model registry logic
│   ├── models.py          # SQLAlchemy database models
│   ├── schema.py          # Pydantic validation schemas
│   ├── main.py            # FastAPI entry point
│   ├── Dockerfile         # Backend service container config
│   └── requirements.txt   # Backend dependencies (TensorFlow, FastAPI, etc.)
├── frontend/
│   ├── app_frontend.py    # Streamlit interactive dashboard
│   ├── Dockerfile         # Frontend service container config
│   └── requirements.txt   # Frontend dependencies (Streamlit, Requests)
├── docker-compose.yml     # Multi-container orchestration
├── .dockerignore          # Optimization to exclude venv and local DB from builds
└── .gitignore             # Ensures large models and venv are not tracked

## 📥 Model Files (Download Links)
Since these model files are too large for GitHub, please download them and place them in the `models_data/` directory:

| File Name | Description | Download Link |
| :--- | :--- | :--- |
| `hybrid_recommender.keras` | Trained Neural Network Model | [Download](https://drive.google.com/file/d/1P1RCwoNYsT7msgrFpcn1h3bhems3-8Bf/view?usp=drive_link) |
| `similarity.pkl` | Pre-computed Similarity Matrix | [Download](https://drive.google.com/file/d/1Mbqax9__L46c5smeIlprxJ23v3C2erJU/view?usp=drive_link) |
| `movies_list.pkl` | Processed Movie Metadata | [Download](https://drive.google.com/file/d/1XVFBtfRhSKa9fgKs4udFxGFpEflRN4dc/view?usp=drive_link) |
| `movie_map.pkl` | Movie ID Mapping File | [Download](https://drive.google.com/file/d/1Q5C86SXC8VIqU2v-Dn6F95e7bysIktAd/view?usp=drive_link) |
| `occupation_map.pkl` | User Occupation Mapping File | [Download](https://drive.google.com/file/d/12x3GU20vLjg0501yOKX4gfemxoNu_EM1/view?usp=drive_link) |
| `max_age.pkl` | Age Normalization Data | [Download](https://drive.google.com/file/d/1uN6vG6CAHcx-flQlNCQqLhYe7g9YvGIh/view?usp=drive_link) |

## 👤 Author
* **Ermia Masoumi**
* GitHub: [basic-member](https://github.com/basic-member)
* LinkedIn: [basic-member](https://www.linkedin.com/in/basic-member3/)