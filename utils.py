import numpy as np
import pandas as pd

USERS_ID = set()

MOVIE_PATH = "C:/Users/aslan/Downloads/archive/movie.csv"

N_PARQUET_FILES = 0

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]
embedding_user_cols = [  'user_id',
                    'count',
                    'mean_rate',
                    'std_rate',
                    'min_rate',
                    'max_rate',
                    'year',
                    'month',
                    'day',
                    'hour'
                ] + list(map(lambda x: f'ratio_{x}', genres)) + ['ratio_other']

_genre_matrix = None



def get_genre_matrix() -> pd.DataFrame:
    """One-hot жанров с индексом movieId: колонки genres + 'other' (число прочих жанров фильма, напр. IMAX)."""
    global _genre_matrix

    if _genre_matrix is None:
        movies = pd.read_csv(MOVIE_PATH, usecols=['movieId', 'genres'])
        dummies = movies['genres'].str.get_dummies('|')
        known = dummies.reindex(columns=genres, fill_value=0)
        other = dummies.drop(columns=genres, errors='ignore').sum(axis=1)
        _genre_matrix = known.assign(other=other).set_index(movies['movieId']).astype(np.int32)
    return _genre_matrix

