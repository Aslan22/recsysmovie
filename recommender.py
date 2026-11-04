import numpy as np
import pandas as pd
from model import BaseModel
from config import MOVIE_PATH
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

class Recommender():
    def __init__(self, user_id: int=1, engine: Engine=None,
                 n_candidates: int=200,
                 random_candidates: bool=False):
        self.engine = engine
        self.user_id = user_id
        self.n_candidates = n_candidates
        self.random_candidates = random_candidates


    def candidates(self, user_id: int=None)->np.ndarray:
        if user_id is None:
            user_id = self.user_id

        candidates_: pd.DataFrame
        if self.random_candidates:
            candidates_ = pd.read_sql('SELECT movie_id FROM movies '
                             'ORDER BY random() '
                             'LIMIT '+str(self.n_candidates), self.engine)
            return np.array(candidates_)
        else:
            return np.array([None])

    def predict_rating(self, user_id: int=None, films: np.ndarray=None):

        if user_id is None:
            user_id = self.user_id




    def rank(self, data: pd.DataFrame, top_k: int=5):
        indx = np.lexsort((data['rating']))[::-1][:top_k]
        return data.iloc[indx, 0]

    def forward(self, user_id):
        cands = self.candidates(user_id=user_id, random_mode=True, n_films=20)
        data_x = self.predict_rating(user_id=user_id, films=cands)
        return self.rank(data_x)
