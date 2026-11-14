import numpy as np
import pandas as pd

import config
import utils
from model import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import feature_engineering
from model import Model

class Recommender():
    def __init__(self, user_id: int=1, engine: Engine=None,
                 n_candidates: int=config.N_CANDIDATES,
                 random_candidates: bool=True, model: Model=None):
        self.engine = engine
        self.user_id = user_id
        self.n_candidates = n_candidates
        self.random_candidates = random_candidates
        self.model = model


    def candidates(self, user_id: int=None)->np.ndarray:
        if user_id is None:
            user_id = self.user_id

        candidates_: pd.DataFrame
        if self.random_candidates:
            candidates_ = pd.read_sql('SELECT movie_id FROM ' + config.TABLE_MOVIES + ' '
                             'ORDER BY random() '
                             'LIMIT '+str(self.n_candidates), self.engine)
            return candidates_['movie_id'].to_numpy()
        else:
            return np.array([None])


    def predict_rating(self, user_id: int=None, films: np.ndarray=None):

        if user_id is None:
            user_id = self.user_id
        x = pd.DataFrame(
            feature_engineering.FeatureEngineering().make_rows(user_id=user_id, movie_ids=films),
        columns=config.FEATS_CATBOOST)
        y = self.model.predict(x)
        return pd.DataFrame({
            'movie_id':films,
            'rating':y
        })

    def rank(self, data: pd.DataFrame, top_k: int = config.TOP_K):
        return data.nlargest(top_k, 'rating')['movie_id']


    def rec(self, user_id: int=None):
        if user_id is None:
            user_id = self.user_id
        cands = self.candidates(user_id=user_id)
        mo_ra = self.predict_rating(user_id=user_id, films=cands)
        return self.rank(mo_ra)
