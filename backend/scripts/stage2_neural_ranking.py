 import tensorflow as tf

import tensorflow_datasets as tfds

import pandas as pd

import numpy as np

import pickle

import matplotlib.pyplot as plt

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


ds = tfds.load("movielens/100k-ratings", split="train")

df = tfds.as_dataframe(ds)


def safe_decode(col):

    if col.dtype == object:

        try:

            return col.str.decode('utf-8')

        except:

            return col

    return col



df['movie_title'] = safe_decode(df['movie_title'])

df['occupation'] = safe_decode(df['user_occupation_text'])


df['movie_id'] = safe_decode(df['movie_id']).astype(int)

df['rating'] = safe_decode(df['user_rating']).astype(float)

df['raw_age'] = safe_decode(df['raw_user_age']).astype(float)

df['gender'] = safe_decode(df['user_gender']).astype(int)


df['occupation_id'] = df['occupation'].astype('category').cat.codes

df['age_normal'] = df['raw_age'] / df['raw_age'].max()


max_age = df['raw_age'].max()


with open("max_age.pkl", "wb") as f:

    pickle.dump(max_age, f)


occupation_map = dict(enumerate(df['occupation'].astype('category').cat.categories))

occupation_map_inv = {v: k for k, v in occupation_map.items()}


with open("occupation_map.pkl", "wb") as f:

    pickle.dump(occupation_map_inv, f)


movie_map = df[['movie_id', 'movie_title']].drop_duplicates().set_index('movie_id')['movie_title'].to_dict()


with open("movie_map.pkl", "wb") as f:

    pickle.dump(movie_map, f)


final_df = df[['movie_id', 'rating', 'age_normal', 'gender', 'occupation_id']].astype(float)


n_movies = int(final_df.movie_id.max() + 1)

n_occupations = int(final_df.occupation_id.max() + 1)


final_df["rating_norm"] = (final_df["rating"] - 1.0) / 4.0


def build_model():


    emb_reg = regularizers.l2(1e-5)


    movie_in = Input(shape=(1,), name="Movie-Input")

    age_in = Input(shape=(1,), name="Age-Input")

    gender_in = Input(shape=(1,), name="Gender-Input")

    occ_in = Input(shape=(1,), name="Occ-Input")


    # embeddings

    m_emb = Flatten()(

        Embedding(n_movies, 16, embeddings_regularizer=emb_reg)(movie_in)

    )


    occ_emb = Flatten()(

        Embedding(n_occupations, 4, embeddings_regularizer=emb_reg)(occ_in)

    )


    # combine features

    features = Concatenate()([

        m_emb,

        age_in,

        gender_in,

        occ_emb

    ])


    x = Dense(32, activation="relu")(features)

    x = Dropout(0.3)(x)

    x = Dense(16, activation="relu")(x)


    output = Dense(1, activation="sigmoid")(x)


    model = Model(

        inputs=[movie_in, age_in, gender_in, occ_in],

        outputs=output

    )


    model.compile(

        optimizer=tf.keras.optimizers.Adam(1e-3),

        loss="mse",

        metrics=["mae"]

    )


    return model


def df_to_dataset(dataframe, shuffle=True, batch_size=128):


    inputs = {

        "Movie-Input": dataframe.movie_id.values,

        "Age-Input": dataframe.age_normal.values,

        "Gender-Input": dataframe.gender.values,

        "Occ-Input": dataframe.occupation_id.values

    }


    ds = tf.data.Dataset.from_tensor_slices(

        (inputs, dataframe.rating_norm.values)

    )


    if shuffle:

        ds = ds.shuffle(len(dataframe), seed=42)


    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


train, test = train_test_split(final_df, test_size=0.2, random_state=42)


model = build_model()


train_ds = df_to_dataset(train)

test_ds = df_to_dataset(test, shuffle=False)


callbacks = [

    EarlyStopping(patience=3, restore_best_weights=True),

    ReduceLROnPlateau(patience=2, factor=0.5)

]


history = model.fit(

    train_ds,

    validation_data=test_ds,

    epochs=30,

    callbacks=callbacks

)


model.save("hybrid_recommender.keras")


from google.colab import files


files.download("hybrid_recommender.keras")

files.download("movie_map.pkl")

files.download("occupation_map.pkl")

files.download("max_age.pkl")