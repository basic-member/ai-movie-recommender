# 🎬 Hybrid Movie Recommendation System

An AI-powered movie recommendation engine that combines **Content-Based filtering** with **Deep Learning** architectures. The system features a robust **FastAPI** backend and an interactive **Streamlit** frontend.

<div align="center">
  <img src="https://github.com/user-attachments/assets/f9ee5386-15b0-4c87-8af5-48b60ea02470" width="700" alt="Movie Recommender Demo">
  <p><i>Live demo of the Hybrid Recommender System in action</i></p>
</div>

---

## 🚀 Key Features
* **Hybrid Engine:** Integrates TensorFlow (Keras) for deep learning-based ranking and Scikit-learn for content similarity.
* **Modern API:** High-performance asynchronous backend built with FastAPI.
* **Interactive UI:** User-friendly dashboard for searching and receiving real-time recommendations.
* **Data Management:** Efficient data handling using SQLAlchemy and SQLite.

---

## 🧠 Technical Highlights (Code Architecture)

### 1. Database Initialization & Schema
The project uses **SQLAlchemy ORM** for database management. 
* **Setup Phase:** The `Base.metadata.create_all` command ensures that the `recommend.db` file is created with the correct tables.
* **Data Integrity:** Pydantic models validate movie data before it is persisted in the SQLite database.

### 2. The Hybrid Recommendation Engine
The system utilizes a two-stage pipeline:
* **Stage 1: Candidate Generation:** Uses `cosine_similarity` on movie metadata to find the top 50 similar candidates.
* **Stage 2: Neural Ranking:** These candidates are fed into a **TensorFlow/Keras** model (`hybrid_recommender.keras`) to predict user ratings and provide the final Top-10 personalized results.

---

## 👤 Author
**Ermia Masoumi**
- GitHub: [@basic-member](https://github.com/basic-member)
- LinkedIn: [Ermia Masoumi](https://www.linkedin.com/in/basic-member3)

## 📥 Model Files (Download Links)
Since these model files are too large for GitHub, please download them and place them in the `models_data/` directory:

| File Name | Description | Download Link |
| :--- | :--- | :--- |
| `hybrid_recommender.keras` | Trained Neural Network Model | [Download](https://drive.google.com/file/d/1Mbqax9__L46c5smeIlprxJ23v3C2erJU/view?usp=sharing) |
| `similarity.pkl` | Pre-computed Similarity Matrix | [Download](https://drive.google.com/file/d/1XVFBtfRhSKa9fgKs4udFxGFpEflRN4dc/view?usp=sharing) |
| `movies_list.pkl` | Processed Movie Metadata | [Download](https://drive.google.com/file/d/1uN6vG6CAHcx-flQlNCQqLhYe7g9YvGIh/view?usp=sharing) |
| `movie_map.pkl` | Movie ID Mapping File | [Download](https://drive.google.com/file/d/12x3GU20vLjg0501yOKX4gfemxoNu_EM1/view?usp=sharing) |
| `occupation_map.pkl` | User Occupation Mapping File | [Download](https://drive.google.com/file/d/1P1RCwoNYsT7msgrFpcn1h3bhems3-8Bf/view?usp=sharing) |
| `max_age.pkl` | Age Normalization Data | [Download](https://drive.google.com/file/d/1Q5C86SXC8VIqU2v-Dn6F95e7bysIktAd/view?usp=sharing) |

---

## 📂 Project Structure
```text
.
├── backend/          # FastAPI logic, Database models, and Schema
├── frontend/         # Streamlit UI implementation
├── models_data/      # Trained .keras models and .pkl data files (Download separately)
├── requirements.txt  # Project dependencies
└── README.md
