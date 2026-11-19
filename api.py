"""
HTTP-API рекомендательного сервиса.

API — это «окно» в ядро рекомендаций для других программ. Интерфейс на Streamlit
(и кто угодно ещё) не импортирует recommender.py напрямую, а отправляет сюда
HTTP-запросы и получает ответы в JSON. Вся логика остаётся в recommender.py,
здесь только приём запросов, проверка входных данных и сборка ответа.

Запуск из корня проекта:

    uvicorn api:app --reload

  api:app   — «в файле api.py взять объект app»;
  --reload  — перезапускать сервер при каждом сохранении файла (только для разработки).

После запуска в браузере открывается документация со всеми адресами,
где каждый можно вызвать кнопкой «Try it out»:

    http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

import config
import model
import recommender


# ---------------------------------------------------------------------------
# 1. Что загружается один раз при запуске сервера
# ---------------------------------------------------------------------------

# Здесь лежат объекты, которые дорого создавать на каждый запрос:
# соединение с базой, модель (десятки мегабайт), справочник фильмов.
# Они создаются в lifespan() и дальше используются всеми запросами.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код до yield выполняется при запуске сервера, после yield — при остановке.

    # pool_pre_ping=True: перед каждым запросом проверять, что соединение живо.
    # Иначе после перезапуска Postgres или долгого простоя первый запрос упадёт.
    engine = create_engine(config.RECSYS_DB_URL, pool_pre_ping=True)

    # Обученная модель из artifacts/models/catboost/catboost.cbm.
    cat_model = model.Model(model=CatBoostRegressor).load_model()

    state['engine'] = engine
    state['rec'] = recommender.Recommender(engine=engine, model=cat_model)

    # Справочник фильмов (около 27 тысяч строк) целиком в памяти:
    # названия и жанры нужны в каждом ответе, ходить за ними в базу незачем.
    movies = pd.read_sql(text(f'SELECT movie_id, title, genres FROM {config.TABLE_MOVIES}'), engine)
    state['movies'] = movies.set_index('movie_id')

    yield

    engine.dispose()
    state.clear()


app = FastAPI(title='Movie recommender', lifespan=lifespan)


# ---------------------------------------------------------------------------
# 2. Схемы: как выглядят данные на входе и на выходе
# ---------------------------------------------------------------------------
# FastAPI по этим классам сам проверяет входящий JSON (и отвечает ошибкой 422,
# если что-то не так), сам превращает ответ в JSON и сам рисует документацию на /docs.

class NewUser(BaseModel):
    user_id: int


class UserInfo(BaseModel):
    user_id: int
    n_ratings: int      # сколько всего оценок: из датасета и из сервиса
    cold_start: bool    # True — пока показываем лучшие фильмы для всех, а не модель


class Movie(BaseModel):
    movie_id: int
    title: str
    genres: list[str]


class Recommendation(Movie):
    # Предсказание модели. На холодном старте модель не вызывается, поэтому None.
    predicted_rating: float | None = None


class Recommendations(BaseModel):
    user_id: int
    mode: str           # 'cold_start' или 'model' — чтобы интерфейс мог это показать
    items: list[Recommendation]


class RatingIn(BaseModel):
    movie_id: int
    # Field задаёт правила проверки: ge — не меньше, le — не больше,
    # multiple_of — кратно шагу. Оценки 7, 0 или 3.7 не дойдут до базы:
    # FastAPI сразу ответит ошибкой 422 с объяснением.
    rating: float = Field(ge=config.RATING_MIN, le=config.RATING_MAX, multiple_of=config.RATING_STEP)


class RatingOut(BaseModel):
    user_id: int
    movie_id: int
    rating: float


class RatedMovie(Movie):
    rating: float
    ts: datetime


# ---------------------------------------------------------------------------
# 3. Вспомогательные функции
# ---------------------------------------------------------------------------

def movie_cards(movie_ids) -> list[dict]:
    """
    По списку movie_id возвращает карточки фильмов: номер, название, жанры.
    Порядок сохраняется. Фильмы, которых нет в справочнике, пропускаются.
    """
    movies = state['movies']
    cards = []
    for mid in movie_ids:
        mid = int(mid)   # numpy-число -> обычное int, иначе не превратится в JSON
        if mid not in movies.index:
            continue
        genres = movies.at[mid, 'genres']
        cards.append({
            'movie_id': mid,
            'title': movies.at[mid, 'title'],
            # В MovieLens жанры записаны строкой через '|'. Отдаём списком.
            'genres': [] if genres == '(no genres listed)' else genres.split('|'),
        })
    return cards


def user_exists(user_id: int) -> bool:
    """Человек есть в датасете или зарегистрирован в сервисе."""
    with state['engine'].connect() as conn:
        return conn.execute(text(
            f'SELECT EXISTS (SELECT 1 FROM {config.TABLE_USERS_NEW} WHERE user_id = :uid) '
            f'    OR EXISTS (SELECT 1 FROM {config.TABLE_RATING} WHERE user_id = :uid)'
        ), {'uid': int(user_id)}).scalar()


def require_user(user_id: int) -> None:
    """Прерывает запрос с ошибкой 404, если такого пользователя нет."""
    if not user_exists(user_id):
        # HTTPException — способ ответить клиенту ошибкой с нужным кодом и текстом.
        # 404 означает «не найдено».
        raise HTTPException(status_code=404, detail=f'Пользователь {user_id} не найден')


def count_ratings(user_id: int) -> int:
    with state['engine'].connect() as conn:
        return conn.execute(text(
            f'SELECT (SELECT COUNT(*) FROM {config.TABLE_RATING}     WHERE user_id = :uid) '
            f'     + (SELECT COUNT(*) FROM {config.TABLE_RATING_NEW} WHERE user_id = :uid)'
        ), {'uid': int(user_id)}).scalar()


def rated_movie_ids(user_id: int) -> np.ndarray:
    """Все фильмы, которые человек уже оценил, из обеих таблиц."""
    with state['engine'].connect() as conn:
        ids = conn.execute(text(
            f'SELECT movie_id FROM {config.TABLE_RATING}     WHERE user_id = :uid '
            f'UNION '
            f'SELECT movie_id FROM {config.TABLE_RATING_NEW} WHERE user_id = :uid'
        ), {'uid': int(user_id)}).scalars().all()
    return np.array(ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# 4. Адреса (endpoints)
# ---------------------------------------------------------------------------
# Функции объявлены через обычный def, а не async def. Внутри них медленные
# синхронные вызовы: запросы к базе и расчёт признаков на несколько секунд.
# Обычные def FastAPI запускает в отдельных потоках, поэтому пока один человек
# ждёт рекомендаций, сервер продолжает отвечать остальным.

@app.get('/health')
def health():
    """Проверка, что сервер запущен. Пригодится Docker, чтобы понять, что API поднялся."""
    return {'status': 'ok'}


@app.post('/users', response_model=NewUser, status_code=201)
def register():
    """Регистрация: выдаёт новый номер пользователя. 201 означает «создано»."""
    return {'user_id': state['rec'].get_new_user_id()}


@app.get('/users/{user_id}', response_model=UserInfo)
def get_user(user_id: int):
    """Вход по номеру: есть ли такой человек и на каком он этапе."""
    require_user(user_id)
    return {
        'user_id': user_id,
        'n_ratings': count_ratings(user_id),
        'cold_start': bool(state['rec'].is_cold_start(user_id)),
    }


@app.get('/users/{user_id}/recommendations', response_model=Recommendations)
def get_recommendations(user_id: int):
    """
    Рекомендации. Здесь та же логика, что в Recommender.rec(), но собранная
    из его отдельных методов, потому что API нужно больше, чем возвращает rec():
      - предсказанная оценка у каждого фильма, а rec() отдаёт только movie_id;
      - режим (холодный старт или модель), чтобы интерфейс мог его показать;
      - уже оценённые фильмы не должны попадать в список.
    """
    require_user(user_id)
    r = state['rec']

    if r.is_cold_start(user_id):
        # Лучшие фильмы для всех, кроме уже оценённых этим человеком.
        ids = r.top_films(config.COLD_START_TOP_N, user_id=user_id)
        return {'user_id': user_id, 'mode': 'cold_start', 'items': movie_cards(ids)}

    # 1) кандидаты, 2) убрать уже оценённые, 3) модель предсказывает оценку каждому,
    # 4) берём TOP_K лучших.
    cands = r.candidates(user_id=user_id)
    cands = cands[~np.isin(cands, rated_movie_ids(user_id))]

    pred = r.predict_rating(user_id=user_id, films=cands)
    top_ids = r.rank(pred)
    pred_by_id = pred.set_index('movie_id')['rating']

    items = movie_cards(top_ids)
    for item in items:
        item['predicted_rating'] = float(pred_by_id[item['movie_id']])
    return {'user_id': user_id, 'mode': 'model', 'items': items}


@app.post('/users/{user_id}/ratings', response_model=RatingOut, status_code=201)
def add_rating(user_id: int, body: RatingIn):
    """
    Поставить оценку. Тело запроса — JSON вида {"movie_id": 318, "rating": 4.5}.
    Повторная оценка того же фильма заменяет старую.
    """
    require_user(user_id)
    if body.movie_id not in state['movies'].index:
        raise HTTPException(status_code=404, detail=f'Фильм {body.movie_id} не найден')

    state['rec'].add_rating(user_id=user_id, movie_id=body.movie_id, rating=body.rating)
    return {'user_id': user_id, 'movie_id': body.movie_id, 'rating': body.rating}


@app.get('/users/{user_id}/ratings', response_model=list[RatedMovie])
def get_history(user_id: int, limit: int = config.HISTORY_LIMIT):
    """
    Последние оценки человека, новые сверху.
    limit — параметр в адресе: /users/5/ratings?limit=50
    """
    require_user(user_id)
    with state['engine'].connect() as conn:
        rows = conn.execute(text(
            f'SELECT movie_id, rating, ts FROM {config.TABLE_RATING}     WHERE user_id = :uid '
            f'UNION ALL '
            f'SELECT movie_id, rating, ts FROM {config.TABLE_RATING_NEW} WHERE user_id = :uid '
            f'ORDER BY ts DESC LIMIT :n'
        ), {'uid': int(user_id), 'n': int(limit)}).all()

    cards = {c['movie_id']: c for c in movie_cards([row.movie_id for row in rows])}
    return [cards[row.movie_id] | {'rating': row.rating, 'ts': row.ts}
            for row in rows if row.movie_id in cards]
