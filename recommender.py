import numpy as np
import pandas as pd
from model import BaseModel
from utils import MOVIE_PATH



def candidates(user_id: int=1, random_mode: bool=False, n_films: int=200):
    movies = pd.read_csv(MOVIE_PATH)
    if random_mode:
        return movies['movieId'].sample(n_films)
    else:
        return None

def predict_rating(user_id: int=1, films: list=None):
    data_x = pd.concat([
        pd.Series(data=np.full(shape=len(films), fill_value=user_id)),
        films
    ], axis=1)
    model = BaseModel(strategy='dummy')

    data_x['rating'] = model.predict(data_x)
    return data_x

def rank(data: pd.DataFrame, top_k: int=5):
    indx = np.lexsort((data['rating']))[::-1][:top_k]
    return data.iloc[indx, 0]

def forward(user_id):
    cands = candidates(user_id=user_id, random_mode=True, n_films=20)
    data_x = predict_rating(user_id=user_id, films=cands)
    return rank(data_x)
