import numpy as np
import pandas as pd

import config
import utils
from model import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import feature_engineering
from model import Model

class Recommender():
    def __init__(self, user_id: int=1, engine: Engine=None,
                 n_candidates: int=config.N_CANDIDATES,
                 random_candidates: bool=True, model: Model=None):

        if engine is None:
            self.engine = create_engine(config.RECSYS_DB_URL)
        else:
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
            feature_engineering.FeatureEngineering(engine=self.engine).make_rows(user_id=user_id, movie_ids=films),
        columns=config.FEATS_CATBOOST)
        y = self.model.predict(x)
        return pd.DataFrame({
            'movie_id':films,
            'rating':y
        })

    def rank(self, data: pd.DataFrame, top_k: int = config.TOP_K):
        return data.nlargest(top_k, 'rating')['movie_id']


    def is_cold_start(self, user_id: int=1):
        with self.engine.begin() as connect:
            d = connect.execute(
                text(f'SELECT '
                     f'(SELECT count(*) FROM {config.TABLE_RATING} '
                     f'WHERE user_id=:uid) + '
                     f'(SELECT count(*) FROM {config.TABLE_RATING_NEW} '
                     f'WHERE user_id=:uid)'), parameters={'uid': user_id}
            ).scalar()

        return d < 5


    def top_films(self, n: int=10, user_id: int=None):

        seen = ''
        params = {'m': config.COLD_START_SMOOTHING, 'n': int(n)}
        if user_id is not None:
            seen = (f'WHERE s.movie_id NOT IN ('
                    f'SELECT movie_id FROM {config.TABLE_RATING} WHERE user_id = :uid '
                    f'UNION ALL '
                    f'SELECT movie_id FROM {config.TABLE_RATING_NEW} WHERE user_id = :uid) ')
            params['uid'] = int(user_id)


        top = pd.read_sql(
            text(f'WITH g AS (SELECT SUM(movie_sum) / SUM(movie_cnt) AS c '
                 f'           FROM {config.TABLE_MOVIE_STATS}) '
                 f'SELECT s.movie_id '
                 f'FROM {config.TABLE_MOVIE_STATS} s CROSS JOIN g '
                 f'{seen}'
                 f'ORDER BY (s.movie_sum + :m * g.c) / (s.movie_cnt + :m) DESC '
                 f'LIMIT :n'),
            con=self.engine, params=params
        )
        return top['movie_id']


    def rec(self, user_id: int=None):
        if user_id is None:
            user_id = self.user_id
        if self.is_cold_start(user_id):
            return self.top_films(n=10, user_id=user_id)

        cands = self.candidates(user_id=user_id)
        mo_ra = self.predict_rating(user_id=user_id, films=cands)
        return self.rank(mo_ra)

    def add_rating(self, user_id: int=1, movie_id: int=1, rating: float=0.0):
        if 0.0 < rating <= 5.0:
            with self.engine.begin() as connect:
                connect.execute(
                    text(f'INSERT INTO {config.TABLE_RATING_NEW} (user_id, movie_id, rating, ts) '
                         f'VALUES (:uid, :mid, :r, now()) '
                         f'ON CONFLICT (user_id, movie_id) DO UPDATE SET rating = EXCLUDED.rating, ts = now()'),
                    parameters={'uid':int(user_id),
                                'mid': int(movie_id),
                                'r':float(rating)}
                )
        else:
            raise ValueError('Рейтинг должен быть положительным и не более 5!')

    def get_new_user_id(self):
        with self.engine.connect() as connect:
            m = connect.execute(
                text(f'SELECT GREATEST('
                     f'(SELECT max(user_id) FROM {config.TABLE_USERS_NEW}), '
                     f'(SELECT max(user_id) FROM {config.TABLE_RATING}))')
            ).scalar()

        with self.engine.begin() as connect:
            connect.execute(text(f'INSERT INTO {config.TABLE_USERS_NEW}(user_id) '
                            f'VALUES(:m)'), parameters={'m':m+1})

        return m + 1


