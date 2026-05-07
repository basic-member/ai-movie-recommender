# 🎬 AI Hybrid Movie Recommendation System

An enterprise-grade, AI-powered movie recommendation engine that harmonizes **Content-Based filtering** with **Deep Learning (Neural Collaborative Filtering)**. This full-stack application features a high-performance **FastAPI** backend and a sleek, interactive **Streamlit** frontend.

<div align="center">
  <img src="https://github.com/user-attachments/assets/f9ee5386-15b0-4c87-8af5-48b60ea02470" width="700" alt="Movie Recommender Demo">
  <p><i>Empowering users with personalized cinematic discoveries</i></p>
</div>

---

## 🌟 Key Features

*   **🧠 Hybrid Intelligence:** Orchestrates a dual-stage pipeline combining **Scikit-learn** (Content-based) and **TensorFlow/Keras** (Neural Ranking).
*   **🔐 Secure Authentication:** Robust **JWT & bcrypt** implementation with dedicated endpoints for registration and login.
*   **👤 Personalization Engine:** Real-time recommendations tailored to user demographics (age, gender, occupation) and interaction history.
*   **❤️ Interaction Loop:** Interactive "Like" mechanism to save preferences and refine the recommendation model.
*   **⚡ Optimized Performance:** Integrated **Cache Management** system to minimize database load and maximize response speed.
*   **📡 TMDB Integration:** Dynamic movie metadata and poster retrieval via The Movie Database (TMDB) API.
*   **🐳 Container Ready:** Fully Dockerized architecture with `docker-compose` for seamless deployment.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, FastAPI, SQLAlchemy, Pydantic, OAuth2 (JWT) |
| **Frontend** | Streamlit, Requests, Pandas |
| **Machine Learning** | TensorFlow (Keras), Scikit-learn, Pickle |
| **Database** | SQLite (ORM managed) |
| **DevOps** | Docker, Docker Compose, Python-Dotenv |

---

## 🧠 Architecture Overview

### 1. The Recommendation Pipeline
The system utilizes a sophisticated two-stage candidate-ranking architecture:
*   **Stage 1: Candidate Generation:** Uses `cosine_similarity` on processed movie metadata to identify the top 50 potential matches.
*   **Stage 2: Neural Ranking:** These candidates, along with the user's profile, are processed by a **Deep Neural Network** (`hybrid_recommender.keras`) to predict ratings and deliver the top-10 personalized results.

### 2. Data Management
*   **SQLAlchemy ORM:** Manages Users, Movies, and User-Movie interaction (Likes) tables.
*   **Auto-Sync:** Built-in administrative tools to synchronize database records with pre-processed movie metadata.

---

## 📥 Model & Data Requirements

The following pre-trained models and data files are required in the `models_data/` directory:

| File Name | Description | Download Link |
| :--- | :--- | :--- |
| `hybrid_recommender.keras` | Neural Ranking Model | [Download](https://drive.google.com/file/d/1Mbqax9__L46c5smeIlprxJ23v3C2erJU/view?usp=sharing) |
| `similarity.pkl` | Similarity Matrix | [Download](https://drive.google.com/file/d/1XVFBtfRhSKa9fgKs4udFxGFpEflRN4dc/view?usp=sharing) |
| `movies_list.pkl` | Movie Metadata | [Download](https://drive.google.com/file/d/1uN6vG6CAHcx-flQlNCQqLhYe7g9YvGIh/view?usp=sharing) |
| `movie_map.pkl` | ID Mapping | [Download](https://drive.google.com/file/d/12x3GU20vLjg0501yOKX4gfemxoNu_EM1/view?usp=sharing) |
| `occupation_map.pkl` | Occupation Data | [Download](https://drive.google.com/file/d/1P1RCwoNYsT7msgrFpcn1h3bhems3-8Bf/view?usp=sharing) |
| `max_age.pkl` | Normalization Data | [Download](https://drive.google.com/file/d/1Q5C86SXC8VIqU2v-Dn6F95e7bysIktAd/view?usp=sharing) |

---

## 🚀 Getting Started

### 1. Configuration
Create a `.env` file in the root directory:
```env
TMDB_API_KEY=your_tmdb_api_key_here
BACKEND_URL=http://localhost:8000
```

### 2. Manual Installation
```bash
# Setup Backend
cd backend
pip install -r requirements.txt
uvicorn api:app --reload

# Setup Frontend (New Terminal)
cd frontend
pip install -r requirements.txt
streamlit run app_frontend.py
```

### 3. Docker Deployment
```bash
docker-compose up --build
```

---

## 📂 Project Structure
```text
.
├── backend/
│   ├── routers/        # Auth, Activities, Admin, Recommender modules
│   ├── models_data/    # ML models and pickles
│   ├── api.py          # Application entry point
│   ├── database.py     # SQLAlchemy configuration
│   └── ml_manager.py   # Hybrid engine logic
├── frontend/
│   └── app_frontend.py # Streamlit UI
├── docker-compose.yml  # Multi-container orchestration
└── README.md
```

---

## 👤 Author
**Ermia Masoumi**
- GitHub: [@basic-member](https://github.com/basic-member)
- LinkedIn: [Ermia Masoumi](https://www.linkedin.com/in/basic-member3)

---
<div align="center">
  <sub>Built with ❤️ for Movie Lovers and AI Enthusiasts</sub>
</div>
