import os
import bz2
import pickle
import tensorflow as tf
import numpy as np
import sys
import traceback

# --------------------------------------------------------------------------
# ENVIRONMENT COMPATIBILITY
# --------------------------------------------------------------------------
# Fix for NumPy 2.x compatibility with older pickles
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models_data")

# --------------------------------------------------------------------------
# INTERNAL PRIVATE CACHE (SINGLETONS)
# --------------------------------------------------------------------------
_movies_df = None
_tf_model = None
_similar_model = None
_max_age = None
_occu_map = None

# --------------------------------------------------------------------------
# LAZY LOADING GETTERS
# --------------------------------------------------------------------------

def get_movies_df():
    """Returns the movies DataFrame for database synchronization."""
    global _movies_df
    if _movies_df is None:
        try:
            print("📂 [ML-Manager] Loading Movies DataFrame...")
            path = os.path.join(MODELS_PATH, "movies_list.pkl")
            with open(path, 'rb') as f:
                _movies_df = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading movies_df: {e}")
    return _movies_df

def get_tf_model():
    """Returns the Keras model. Only initialized when first recommendation is requested."""
    global _tf_model
    if _tf_model is None:
        try:
            print("🤖 [ML-Manager] Initializing Heavy TensorFlow Engine...")
            path = os.path.join(MODELS_PATH, "hybrid_recommender.keras")
            _tf_model = tf.keras.models.load_model(path)
        except Exception as e:
            print(f"❌ Error loading TensorFlow model: {e}")
    return _tf_model

def get_similarity_matrix():
    """Returns the compressed similarity matrix."""
    global _similar_model
    if _similar_model is None:
        try:
            print("📂 [ML-Manager] Decompressing Similarity Matrix (bz2)...")
            path = os.path.join(MODELS_PATH, "similarity.pbz2")
            with bz2.BZ2File(path, 'rb') as f:
                _similar_model = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading similarity matrix: {e}")
    return _similar_model

def get_max_age():
    """Returns the normalization factor for user age."""
    global _max_age
    if _max_age is None:
        try:
            path = os.path.join(MODELS_PATH, "max_age.pkl")
            with open(path, 'rb') as f:
                _max_age = pickle.load(f)
        except:
            _max_age = 1  # Fallback to 1 to avoid division by zero
    return _max_age

def get_occu_map():
    """Returns the occupation mapping for user profiling."""
    global _occu_map
    if _occu_map is None:
        try:
            path = os.path.join(MODELS_PATH, "occupation_map.pkl")
            with open(path, 'rb') as f:
                _occu_map = pickle.load(f)
        except:
            _occu_map = {}
    return _occu_map