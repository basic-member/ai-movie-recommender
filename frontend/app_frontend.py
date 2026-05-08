import streamlit as st
import requests
import os
import pickle
from dotenv import load_dotenv

# Try to load .env, but don't fail if it's missing (env vars are better for production)
env_path = os.path.join(os.path.dirname(__file__), "../.env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Also try the current directory just in case
    load_dotenv()

# --- CONFIGURATION ---
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

if not TMDB_API_KEY:
    st.error("🔑 **TMDB API Key missing!** Set it in environment variables or .env file.")

st.set_page_config(page_title="AI Hybrid Movie Recommender", page_icon="🎬", layout="wide")

# --- SESSION STATE ---
if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "guest_data" not in st.session_state:
    st.session_state.guest_data = {"age": 20, "gender": "Male", "occu": "Other"}

# --- UTILS ---
@st.cache_data(show_spinner=False)
def get_poster(tmdb_id):
    if not tmdb_id or tmdb_id == 0:
        return "https://via.placeholder.com/500x750?text=No+ID"
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        path = data.get('poster_path')
        if path:
            return f"https://image.tmdb.org/t/p/w500{path}"
    except:
        pass
    return f"https://via.placeholder.com/500x750?text=Error+ID:{tmdb_id}"

@st.cache_data(show_spinner=False)
def get_all_movies():
    try:
        response = requests.get(f"{BACKEND_URL}/setup-db/list", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return []

@st.cache_data
def get_occupations():
    try:
        response = requests.get(f"{BACKEND_URL}/recommender/occupations", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return ["Student", "Engineer", "Technician", "Other"]

def like_movie(movie_id):
    if not st.session_state.token:
        return {"message": "Please login first"}
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        res = requests.post(f"{BACKEND_URL}/movies/{movie_id}/like", headers=headers)
        return res.json()
    except:
        return {"message": "Connection Error"}

all_occupations = get_occupations()

# --- SIDEBAR ---
st.sidebar.title("🎬 AI Recommender")

if st.session_state.token is None:
    tab_login, tab_signup, tab_guest = st.sidebar.tabs(["🔑 Login", "📝 Signup", "👤 Guest"])
    with tab_login:
        l_email = st.text_input("Email", key="l_email")
        l_password = st.text_input("Password", type="password", key="l_pass")
        if st.button("Sign In"):
            res = requests.post(f"{BACKEND_URL}/auth/login", json={"email": l_email, "password": l_password})
            if res.status_code == 200:
                data = res.json()
                st.session_state.token = data["access_token"]
                st.session_state.user_email = data["email"]
                st.rerun()
            else:
                st.error("Login Failed")
    with tab_signup:
        s_email = st.text_input("Email", key="s_email")
        s_password = st.text_input("Password", type="password", key="s_pass")
        s_name = st.text_input("Name")
        s_age = st.number_input("Age", 5, 100, 20)
        s_gender = st.selectbox("Gender", ["Male", "Female"])
        s_occu = st.selectbox("Occupation", all_occupations)
        if st.button("Create Account"):
            p = {"email": s_email, "password": s_password, "name": s_name, "age": s_age, "gender": s_gender, "occu": s_occu}
            res = requests.post(f"{BACKEND_URL}/auth/register", json=p)
            if res.status_code == 200:
                data = res.json()
                st.session_state.token = data["access_token"]
                st.session_state.user_email = data["email"]
                st.rerun()
    with tab_guest:
        g_age = st.number_input("Guest Age", 5, 100, 20)
        g_gender = st.selectbox("Guest Gender", ["Male", "Female"])
        g_occu = st.selectbox("Guest Occupation", all_occupations)
        if st.button("Update Guest Profile"):
            st.session_state.guest_data = {"age": g_age, "gender": g_gender, "occu": g_occu}
            st.toast("Profile updated!")
else:
    st.sidebar.success(f"User: {st.session_state.user_email}")
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.rerun()

# --- MAIN ---
st.title("🎬 AI-Powered Hybrid Recommender")

movie_list = get_all_movies()

if movie_list is None:
    st.error("Backend offline.")
    if st.button("Retry"): st.rerun()
elif len(movie_list) == 0:
    st.warning("Database empty.")
    if st.button("Sync Database"):
        requests.get(f"{BACKEND_URL}/setup-db")
        st.rerun()
else:
    # Top Section: Selection and Main Movie Like
    movie_dict = {m['title']: (m['id'], m.get('tmdb_id', 0)) for m in movie_list}
    col_sel, col_lk = st.columns([4, 1])
    
    with col_sel:
        selected_title = st.selectbox("Select a Movie:", list(movie_dict.keys()))
        target_id, target_tmdb = movie_dict[selected_title]
    
    with col_lk:
        st.write("") # spacing
        if st.session_state.token:
            if st.button("❤️ Like This", help="Add to your preferences"):
                msg = like_movie(target_id)
                st.toast(msg['message'])
        else:
            st.caption("Login to Like")

    if st.button("🚀 Generate Recommendations", use_container_width=True):
        headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
        payload = None if st.session_state.token else st.session_state.guest_data
        
        with st.spinner("Calculating..."):
            res = requests.post(f"{BACKEND_URL}/recommender/recommend/hybrid?movie_id={target_id}", json=payload, headers=headers)
            if res.status_code == 200:
                results = res.json()
                st.divider()
                st.subheader(f"Recommendations based on '{selected_title}':")
                
                cols = st.columns(5)
                for i, movie in enumerate(results):
                    with cols[i]:
                        # Debug: Show the ID for a moment to see if it exists
                        t_id = movie.get('tmdb_id')
                        # Poster
                        st.image(get_poster(t_id), use_container_width=True)
                        # Title
                        st.write(f"**{movie['title']}**")
                        # Progress Bar & Match %
                        match_pct = int(movie['score'] * 100)
                        st.progress(match_pct / 100)
                        st.caption(f"Match: {match_pct}%")
                        # Like Button
                        if st.session_state.token:
                            if st.button("Like", key=f"rec_{movie['id']}"):
                                f = like_movie(movie['id'])
                                st.toast(f['message'])
            else:
                st.error("Engine Error")