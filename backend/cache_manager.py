class RecommendationCache:
    def __init__(self):
        self._cache = {}

    def get_key(self, movie_id, user_id=None, guest_data=None):
        # Create a unique key based on movie and user/guest profile
        if user_id:
            return f"user_{user_id}_movie_{movie_id}"
        if guest_data:
            # Sort keys to ensure consistent hashing
            profile_str = "_".join([f"{k}:{v}" for k, v in sorted(guest_data.items())])
            return f"guest_{profile_str}_movie_{movie_id}"
        return f"movie_{movie_id}"

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value):
        # We can limit cache size here if needed
        if len(self._cache) > 1000:
            self._cache.clear() # Simple flush if too big
        self._cache[key] = value

    def invalidate_user(self, user_id):
        # Remove all entries for a specific user when they like something
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"user_{user_id}_")]
        for k in keys_to_remove:
            del self._cache[k]

# Singleton instance
cache_instance = RecommendationCache()
