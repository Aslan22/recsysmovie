import numpy as np
import pandas as pd
import utils
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.types import Integer, Float, DateTime, REAL
import pathlib
import os
from dotenv import load_dotenv

def _exclusive_group_cumsum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    excl = np.cumsum(values, axis=0) - values
    return excl - excl[starts]

import csv
from io import StringIO

def psql_insert_copy(table, conn, keys, data_iter):
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        buf = StringIO()
        csv.writer(buf).writerows(data_iter)
        buf.seek(0)
        columns = ', '.join(f'"{k}"' for k in keys)
        name = f'{table.schema}.{table.name}' if table.schema else table.name
        cur.copy_expert(f'COPY {name} ({columns}) FROM STDIN WITH CSV', buf)

class FeatureEngineering():
    def __init__(self, table_name_in: str='rating_cut', user_indx: tuple=(1, 140000)):
        load_dotenv()
        self.database_url = os.environ['RECSYS_DB_URL']
        self.user_indx = user_indx
        self.table_name_in = table_name_in
        self.engine = create_engine(self.database_url)

    def chunk_transform(self, engine, user_ids: tuple=(1, 140000), save_mode: bool=False, table_name: str='rating_upd', load_option: str='sql', n_chunk: int=0):
        l, r = user_ids

        data = pd.read_sql('SELECT * '
                           'FROM '+self.table_name_in+
                           ' WHERE'
                           ' user_id BETWEEN ' + str(l) + ' and ' + str(r), engine)
        data['ts'] = pd.to_datetime(data['ts'])

        x, y = data.drop(columns=['rating']), data['rating']
        x_transform = self.transform(x, y)
        data_transform = pd.concat([x_transform, y], axis=1)
        del data

        if save_mode:
            if load_option == 'sql':
                inspect_ = inspect(engine)
                tables = inspect_.get_table_names(schema="public")

                if table_name not in tables:
                    if_exists = 'fail'
                else:
                    if_exists = 'append'


                data_transform.to_sql(
                    if_exists=if_exists,
                    index=False,
                    name=table_name,
                    con=engine,
                    method=psql_insert_copy
                )
            elif load_option == 'parquet':
                data_transform.to_parquet(f'C:/Users/aslan/Downloads/archive/rating_upd/{n_chunk}.parquet')




        return x_transform


    def forward(self, cut_size: int=1000, delete_bd: bool=True,
                table_name: str='rating_upd', load_option: str='sql'):



        if delete_bd:
            if load_option == 'sql':
                with self.engine.begin() as conn:
                    conn.execute(text("DROP TABLE IF EXISTS "+table_name))
            else:
                utils.N_PARQUET_FILES = 0

                pth = pathlib.Path('C:/Users/aslan/Downloads/archive/rating_upd')
                fls = pth.glob('*.parquet')
                for f in fls:
                    f.unlink()


        start, end = self.user_indx[0], self.user_indx[1]
        n_chunk = 0

        for i in range(start, end, cut_size):
            l, r = i, min(i + cut_size - 1, end)
            self.chunk_transform(engine=self.engine, user_ids=(l, r), save_mode=True,
                                 table_name=table_name, load_option=load_option, n_chunk=n_chunk)
            n_chunk += 1

        utils.N_PARQUET_FILES = n_chunk





    def transform(self, x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:

        user = x['user_id'].to_numpy()
        movie = x['movie_id'].to_numpy()
        ts = x['ts']
        rating = np.asarray(y, dtype=np.float64)


        order = np.lexsort((ts, user))

        user, movie, rating = user[order], movie[order], rating[order]

        n = len(user)
        is_start = np.ones(n, dtype=bool)
        is_start[1:] = user[1:] != user[:-1]
        starts = np.maximum.accumulate(np.where(is_start, np.arange(n), 0))

        cnt = np.arange(n) - starts
        cnt_nan = np.where(cnt > 0, cnt, np.nan)

        mean = _exclusive_group_cumsum(rating, starts) / cnt_nan
        mean_sq = _exclusive_group_cumsum(rating ** 2, starts) / cnt_nan
        std = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))

        r = pd.Series(rating)
        min_rate = r.groupby(user).cummin().groupby(user).shift().to_numpy()
        max_rate = r.groupby(user).cummax().groupby(user).shift().to_numpy()

        gm = utils.get_genre_matrix()
        g = gm.reindex(movie, fill_value=0).to_numpy(np.int32)
        ratios = (_exclusive_group_cumsum(g, starts) / cnt_nan[:, None]).astype(np.float32)

        res = pd.DataFrame({
            'user_id': user,
            'count': cnt,
            'mean_rate': mean,
            'std_rate': std,
            'min_rate': min_rate,
            'max_rate': max_rate,
            'movie_id': movie
        })
        res[[f'ratio_{c}' for c in gm.columns]] = ratios


        res.index = order
        res = res.sort_index()
        res.index = x.index
        res['ts'] = ts
        res['year'] = ts.dt.year
        res['month'] = ts.dt.month
        res['day'] = ts.dt.day
        res['hour'] = ts.dt.hour


        return res.join(gm, how='left', on='movie_id', rsuffix='_is_genre')

    def get_user_stats(self, user_id: int=1, ts: pd.Timestamp=None) -> dict:
        if ts is None:
            ts = pd.Timestamp.now()
        data = pd.read_sql(text('SELECT * FROM :table WHERE user_id=:uid'),
                           self.engine, params={'uid':user_id, 'table':self.table_name_in})
        movie_id = data['movie_id'].to_numpy()
        rating = data['rating'].to_numpy()
        dr = utils.get_genre_matrix().reindex(index=movie_id, fill_value=0).sum() / len(movie_id)
        ratios = {f'ratio_{x}':y for x, y in zip(dr.index, dr)}
        return {
            'count': len(movie_id),
            'mean_rate': np.mean(rating),
            'std_rate': np.std(rating),
            'min_rate': np.min(rating),
            'max_rate': np.max(rating)
        } | ratios | {
            'ts': ts,
            'year': ts.year,
            'month': ts.month,
            'day': ts.day,
            'hour': ts.hour

        }



