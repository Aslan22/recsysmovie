import os
from dotenv import load_dotenv

load_dotenv()

MOVIE_PATH = os.environ['MOVIE_PATH']
RECSYS_DB_URL = os.environ['RECSYS_DB_URL']

N_PARQUET_FILES = 0
TABLE_RATING = 'rating'
TABLE_RATING_CUT = 'rating_cut'
TABLE_RATING_UPD = 'rating_upd'
TABLE_MOVIES = 'movies'