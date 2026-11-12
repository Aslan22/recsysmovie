"""
Все настройки проекта в одном месте.

Правило, по которому значения разложены:

  .env       — то, что отличается от машины к машине или секретно:
               пароль от базы, пути к данным на диске. Файл .env в git не попадает
               (он в .gitignore), образец лежит в .env.example.

  config.py  — константы самого проекта: имена таблиц, имена файлов, размеры чанков,
               параметры моделей. Они одинаковы у всех, кто запускает проект,
               поэтому лежат в коде и попадают в git.

Остальные модули ничего не читают из окружения сами, а берут значения отсюда:

    import config
    config.TABLE_RATING
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Читает файл .env и кладёт его строки в переменные окружения (os.environ).
# Достаточно вызвать один раз здесь: config импортируется раньше остальных модулей.
load_dotenv()


# ---------------------------------------------------------------------------
# 1. Из .env: секреты и пути, которые зависят от компьютера
# ---------------------------------------------------------------------------

# Строка подключения к Postgres. В ней пароль, поэтому она только в .env.
# os.environ[...] (а не os.getenv) выбран намеренно: если переменной нет,
# программа сразу упадёт с понятным KeyError, а не позже с непонятной ошибкой.
RECSYS_DB_URL = os.environ['RECSYS_DB_URL']

# Полный путь к movie.csv из датасета MovieLens.
MOVIE_PATH = os.environ['MOVIE_PATH']

# Папка с датасетом. Необязательная переменная: если DATA_DIR в .env не задана,
# берётся папка, в которой лежит movie.csv.
DATA_DIR = Path(os.getenv('DATA_DIR', Path(MOVIE_PATH).parent))

# Файлы и папки внутри папки с датасетом.
RATING_CUT_CSV = DATA_DIR / 'rating_cut.csv'   # урезанная выборка оценок для экспериментов
PARQUET_DIR = DATA_DIR / 'rating_upd'          # сюда forward() пишет чанки с признаками


# ---------------------------------------------------------------------------
# 2. Пути внутри проекта
# ---------------------------------------------------------------------------

# Папка, где лежит этот файл, то есть корень проекта. Считается автоматически,
# поэтому проект можно положить в любое место на любом компьютере.
PROJECT_DIR = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_DIR / 'artifacts' / 'models'   # обученные модели и их гиперпараметры
REPORTS_DIR = PROJECT_DIR / 'reports'               # метрики

# Ключи словарей ниже — это model_type из model.py:
# 'base', 'catboost', 'lgbm', 'xgb'.

# Куда сохраняется обученная модель.
MODEL_FILES = {
    'base': MODELS_DIR / 'base_model' / 'base.joblib',
    'catboost': MODELS_DIR / 'catboost' / 'catboost.cbm',
    'lgbm': MODELS_DIR / 'lightgbm' / 'lightgbm.txt',
    'xgb': MODELS_DIR / 'xgboost' / 'xgboost.json',
}

# Куда сохраняются лучшие гиперпараметры после подбора через Optuna.
PARAMS_FILES = {
    'base': MODELS_DIR / 'base_model' / 'base_params.json',
    'catboost': MODELS_DIR / 'catboost' / 'catboost_params.json',
    'lgbm': MODELS_DIR / 'lightgbm' / 'lightgbm_params.json',
    'xgb': MODELS_DIR / 'xgboost' / 'xgboost_params.json',
}

# Куда сохраняются метрики. Для неизвестного типа модели используется OTHER_METRICS_FILE.
METRICS_FILES = {
    'base': REPORTS_DIR / 'basemodel_metrics.json',
    'catboost': REPORTS_DIR / 'catboost_metrics.json',
    'lgbm': REPORTS_DIR / 'lgbm_metrics.json',
    'xgb': REPORTS_DIR / 'xgb_metrics.json',
}
OTHER_METRICS_FILE = REPORTS_DIR / 'other_metrics.json'


# ---------------------------------------------------------------------------
# 3. Таблицы в базе
# ---------------------------------------------------------------------------

TABLE_RATING = 'rating'            # все оценки (около 20 млн строк)
TABLE_RATING_CUT = 'rating_cut'    # урезанная выборка (250 тыс. строк)
TABLE_RATING_UPD = 'rating_upd'    # оценки вместе с посчитанными признаками
TABLE_MOVIES = 'movies'            # справочник фильмов
TABLE_MOVIE_FEATS = 'movie_feats'  # накопительные статистики фильма на момент каждой оценки


# ---------------------------------------------------------------------------
# 4. Расчёт признаков (feature_engineering.py)
# ---------------------------------------------------------------------------

# Диапазон user_id, который обрабатывает forward(). Обе границы включаются.
USER_ID_RANGE = (1, 140000)

# Сколько пользователей в одном чанке. Больше — быстрее, но нужно больше памяти.
CHUNK_SIZE = 1000

# Сколько parquet-файлов записал последний запуск forward().
# Это не настройка, а счётчик: forward() сам его обновляет.
N_PARQUET_FILES = 0


# ---------------------------------------------------------------------------
# 5. Модели (model.py)
# ---------------------------------------------------------------------------

# Зерно генератора случайных чисел: с ним подбор гиперпараметров
# и случайные выборки дают одинаковый результат при каждом запуске.
RANDOM_SEED = 42

# Сколько вариантов гиперпараметров пробует Optuna.
N_TRIALS = 50

# Оценка, которую базовая модель предсказывает, когда о пользователе ничего не известно.
DEFAULT_RATING = 3.0


# ---------------------------------------------------------------------------
# 6. Рекомендации (recommender.py)
# ---------------------------------------------------------------------------

N_CANDIDATES = 200   # сколько фильмов-кандидатов отбирается перед ранжированием
TOP_K = 5            # сколько фильмов попадает в итоговую рекомендацию

# ---------------------------------------------------------------------------
# 7. Наименования признаков
# ---------------------------------------------------------------------------
COLUMN_RATING_UPDATE = ['user_id', 'count', 'mean_rate', 'std_rate', 'min_rate', 'max_rate',
       'movie_id', 'ratio_Action', 'ratio_Adventure', 'ratio_Animation',
       'ratio_Children', 'ratio_Comedy', 'ratio_Crime', 'ratio_Documentary',
       'ratio_Drama', 'ratio_Fantasy', 'ratio_Film-Noir', 'ratio_Horror',
       'ratio_Musical', 'ratio_Mystery', 'ratio_Romance', 'ratio_Sci-Fi',
       'ratio_Thriller', 'ratio_War', 'ratio_Western', 'ratio_other',
       'mean_Action', 'mean_Adventure', 'mean_Animation', 'mean_Children',
       'mean_Comedy', 'mean_Crime', 'mean_Documentary', 'mean_Drama',
       'mean_Fantasy', 'mean_Film-Noir', 'mean_Horror', 'mean_Musical',
       'mean_Mystery', 'mean_Romance', 'mean_Sci-Fi', 'mean_Thriller',
       'mean_War', 'mean_Western', 'mean_other', 'ts', 'year', 'month', 'day',
       'hour', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
       'other', 'movie_cnt', 'av_rating_movie', 'rating']

FEATS_CATBOOST = ['count', 'mean_rate', 'std_rate', 'min_rate', 'max_rate', 'ratio_Action', 'ratio_Adventure', 'ratio_Animation',
       'ratio_Children', 'ratio_Comedy', 'ratio_Crime', 'ratio_Documentary',
       'ratio_Drama', 'ratio_Fantasy', 'ratio_Film-Noir', 'ratio_Horror',
       'ratio_Musical', 'ratio_Mystery', 'ratio_Romance', 'ratio_Sci-Fi',
       'ratio_Thriller', 'ratio_War', 'ratio_Western', 'ratio_other',
       'mean_Action', 'mean_Adventure', 'mean_Animation', 'mean_Children',
       'mean_Comedy', 'mean_Crime', 'mean_Documentary', 'mean_Drama',
       'mean_Fantasy', 'mean_Film-Noir', 'mean_Horror', 'mean_Musical',
       'mean_Mystery', 'mean_Romance', 'mean_Sci-Fi', 'mean_Thriller',
       'mean_War', 'mean_Western', 'mean_other', 'month',
       'hour', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
       'other', 'movie_cnt', 'av_rating_movie']