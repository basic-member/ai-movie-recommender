import streamlit as st
import requests
import pickle
from dotenv import load_dotenv
import os

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AI Hybrid Movie Recommender", page_icon="🎬", layout="wide")

# 2. TMDB POSTER FETCHER
def get_poster(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=2)
        data = response.json()
        if data.get('poster_path'):
            return f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

# 3. BACKEND DATA LOADING
@st.cache_data(show_spinner=False)
def get_all_movies():
    try:
        # Increase timeout to give backend time to respond
        response = requests.get("http://127.0.0.1:8000/setup-db/list", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        return None # Return None to indicate a connection error
    return []

@st.cache_data
def get_occupations():
    try:
        with open("../models_data/occupation_map.pkl", 'rb') as f:
            occu_map = pickle.load(f)
        return sorted(list(occu_map.keys()))
    except:
        return ["Student", "Engineer", "Technician", "Other"]
    
all_occupations = get_occupations()

# 4. SIDEBAR - USER PROFILE PERSONALIZATION
st.sidebar.header("👤 User Profile")

user_age = st.sidebar.number_input("Enter Age", min_value=5, max_value=100, value=15, step=1)
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
user_occu = st.sidebar.selectbox("Occupation", all_occupations)

# 5. MAIN INTERFACE
st.title("🎬 AI-Powered Hybrid Recommender")

movie_list = get_all_movies()

# --- CRITICAL FIX: DO NOT HIDE UI IF LIST IS EMPTY ---
if movie_list is None:
    st.error("🔌 **Connection Error:** Cannot reach the Backend. Is your FastAPI server running on port 8000?")
    if st.button("🔄 Retry Connection"):
        st.cache_data.clear()
        st.rerun()
    target_id = None
elif len(movie_list) == 0:
    st.warning("⚠️ **Database Empty:** Please run `http://127.0.0.1:8000/setup-db` in your browser first.")
    target_id = st.number_input("Manual Movie ID (for debugging):", value=0)
    selected_title = "Manual Input"
else:
    st.success(f"✅ {len(movie_list)} Movies Loaded Successfully!")
    movie_dict = {m['title']: m['id'] for m in movie_list}
    selected_title = st.selectbox("Type or Select a Movie:", list(movie_dict.keys()))
    target_id = movie_dict[selected_title]

# 6. RECOMMENDATION BUTTON (Always visible for testing)
if st.button("🚀 Generate Recommendations"):
    if target_id is None:
        st.error("Please fix the connection or enter an ID first.")
    else:
        payload = {"age": user_age, "gender": user_gender, "occu": user_occu}
        with st.spinner("Processing..."):
            try:
                res = requests.post(f"http://127.0.0.1:8000/recommend/hybrid?movie_id={target_id}", json=payload)
                if res.status_code == 200:
                    results = res.json()
                    st.subheader(f"Recommendations based on '{selected_title}'")
                    cols = st.columns(5)
                    for i, movie in enumerate(results):
                        with cols[i]:
                            st.image(get_poster(movie['tmdb_id']), use_container_width=True)
                            st.markdown(f"**{movie['title']}**")
                            score = int(movie['score'] * 100)
                            st.write(f"Match: `{score}%`")
                            st.progress(score / 100)
                else:
                    st.error(f"Error {res.status_code}: {res.json().get('detail')}")
            except:
                st.error("Failed to fetch recommendations.")