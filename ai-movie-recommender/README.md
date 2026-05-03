# 🎬 AI-Powered Hybrid Movie Recommender

An enterprise-grade movie recommendation engine built with **FastAPI** and **TensorFlow**. This system features a sophisticated two-stage pipeline (Retrieval & Ranking) and is fully containerized for high-performance deployment.

## 🧠 Two-Stage AI Architecture

The system utilizes a professional **Two-Stage Pipeline** to provide highly personalized recommendations:

### 1. Stage 1: Candidate Generation (Retrieval)
*   **Model:** Content-Based Filtering (Cosine Similarity).
*   **Logic:** Analyzes movie metadata (genres, overviews) using `CountVectorizer`. Genres are weighted 3x higher to ensure category relevance.
*   **Goal:** Quickly narrows down thousands of movies to the top 30 most relevant candidates.
*   **Script:** `backend/scripts/stage1_content_based.py`

### 2. Stage 2: Neural Re-ranking (Scoring)
*   **Model:** Neural Collaborative Filtering (Deep Learning).
*   **Logic:** A TensorFlow model trained on the MovieLens 100k dataset. It takes the 30 candidates and processes them through an Embedding layer alongside **User Demographics** (Age, Gender, Occupation).
*   **Goal:** Predicts the specific rating a unique user would give to each candidate, delivering a final Top-5 personalized list.
*   **Script:** `backend/scripts/stage2_neural_ranking.py`

## 🚀 Key Engineering Highlights

*   **Modular API Design:** Decoupled backend logic using FastAPI `APIRouter`.
*   **Lazy Loading Engine:** `ml_manager.py` ensures heavy TensorFlow models are loaded only upon the first request to optimize RAM.
*   **Security & Auth:** Implemented OAuth2 with JWT for secure user activities and admin actions.
*   **Full Dockerization:** Multi-container orchestration using `docker-compose`.

## 📂 Project Structure
```text
.
├── backend/
│   ├── models_data/       # AI Models & Data Assets (.keras, .pkl)
│   ├── routers/           # Modular API endpoints (activities, admin, recommender)
│   ├── scripts/           # Core training & logic for Stage 1 & Stage 2
│   ├── ml_manager.py      # Lazy loading & Model registry logic
│   ├── database.py        # SQLAlchemy configuration
│   ├── models.py          # SQLAlchemy database models
│   ├── main.py            # FastAPI entry point
│   └── Dockerfile         # Backend service container config
├── frontend/
│   ├── app_frontend.py    # Streamlit interactive dashboard
│   └── Dockerfile         # Frontend service container config
├── docker-compose.yml     # Multi-container orchestration
└── .gitignore             # Ensures large models and venv are not tracked

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