import kagglehub
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
import os

print(os.listdir(path))


import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

file_path = os.path.join(path, 'tmdb_5000_movies.csv')

data = pd.read_csv(file_path)
df = data[['id', 'title', 'overview', 'genres']].copy()

def convert_genres(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return " ".join(L)

df['genres'] = df['genres'].apply(convert_genres)
df['overview'] = df['overview'].fillna('')

df['tags'] = (df['genres'] + " ") * 3 + df['overview']
df['tags'] = df['tags'].apply(lambda x: x.lower())

df.drop(['genres', 'overview'], axis=1, inplace=True)

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vector)

def recommend(movie_title):
    try:
        index = df[df['title'] == movie_title].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        for i in distances[1:6]:
            print(df.iloc[i[0]].title)
    except:
        print("Movie not found!")

pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(df, open('movies_list.pkl', 'wb'))

# test
recommend('Avatar')

from google.colab import files

files.download("similarity.pkl")
files.download("movies_list.pkl")