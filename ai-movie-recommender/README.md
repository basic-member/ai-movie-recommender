# 🎬 Hybrid Movie Recommendation System 

An AI-powered movie recommendation engine that combines Content-Based filtering with Deep Learning architectures. The system features a robust FastAPI backend and an interactive Streamlit frontend.

### 📺 Project Demo
[![Hybrid Recommender Demo](https://img.shields.io/badge/🎥-View_Live_Demo-red?style=for-the-badge)](https://github.com/user-attachments/assets/d4b5825b-c28d-4e1a-a8c7-b99ba06c759c)

## 🚀 Key Features 
* **Hybrid Engine:** Integrates TensorFlow (Keras) for deep learning-based ranking and Scikit-learn for content similarity. 
* **Modern API:** High-performance asynchronous backend built with FastAPI. 
* **Interactive UI:** User-friendly dashboard for searching and receiving real-time recommendations. 
* **Data Management:** Efficient data handling using SQLAlchemy and SQLite. 

## 🧠 Technical Highlights (Code Architecture) 

### 1. Database Initialization & Schema 
The project uses SQLAlchemy ORM for database management. 
* **Setup Phase:** The `Base.metadata.create_all` command ensures that the `recommend.db` file is created with the correct tables. 
* **Data Integrity:** Pydantic models validate movie data before it is persisted in the SQLite database. 

### 2. The Hybrid Recommendation Engine 
The system utilizes a two-stage pipeline: 
* **Stage 1: Candidate Generation:** Uses `cosine_similarity` on movie metadata to find the top 50 similar candidates. 
* **Stage 2: Neural Ranking:** These candidates are fed into a TensorFlow/Keras model (`hybrid_recommender.keras`) to predict user ratings and provide the final Top-10 personalized results. 

## 🔐 Environment Variables (.env) Setup
For security and production best practices, this project uses a `.env` file to manage sensitive credentials and configurations.

1. **Create a file** named `.env` in the root directory.
2. **Add your credentials** in the following format:

```text
# Anthropic API Key for Agentic Reasoning
ANTHROPIC_API_KEY=your_actual_api_key_here

# Database connection string
DATABASE_URL=sqlite:///./recommend.db

> **Security Note:** The `.env` file is included in `.gitignore` to prevent your credentials from being leaked to GitHub.

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

## 📂 Project Structure
```text
. 
├── backend/          # FastAPI logic, Database models, and Schema 
├── frontend/         # Streamlit UI implementation 
├── models_data/      # Trained .keras models and .pkl data files 
├── requirements.txt  # Project dependencies 
└── README.md