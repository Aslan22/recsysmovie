from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import optuna
import json
from typing import Callable
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error as RMSE, mean_absolute_error as MAE, mean_squared_error as MSE, mean_absolute_percentage_error as MAPE
import joblib
from pathlib import Path
import config
import lightgbm


class BaseModel():
    def __init__(self, strategy: str='mean', config_value: float=config.DEFAULT_RATING):
        self.strategy = strategy
        self.model = None
        self.config_value = config_value


    def fit(self, x: pd.DataFrame, y: pd.Series):
        if self.strategy == 'mean':
            self.model = dict(pd.concat([x, y], axis=1).groupby(by='user_id')['rating'].mean())
        elif self.strategy == 'constant':
            self.model = self.config_value
        elif self.model == 'dummy':
            self.model = 'dummy'
        return self

    def predict(self, x: pd.DataFrame):
        if self.strategy == 'mean':
            us_id = x['user_id']
            return np.array(us_id.map(self.model).fillna(self.config_value))
        elif self.strategy == 'constant':
            return np.full(len(x), self.config_value)
        elif self.strategy == 'dummy':
            return np.random.uniform(low=2.5, high=5, size=len(x))
        else:
            return None


def _model_type(model_cls) -> str | None:

    if model_cls is BaseModel:
        return 'base'
    if model_cls is CatBoostRegressor:
        return 'catboost'
    if model_cls is LGBMRegressor:
        return 'lgbm'
    if model_cls is XGBRegressor:
        return 'xgb'
    return None


class Tuner():
    def __init__(self, model: type[BaseModel]
                              |type[CatBoostRegressor]
                              |type[LGBMRegressor]
                              |type[XGBRegressor], direction: str='minimize',
                 metric: Callable=MSE, n_trials: int=config.N_TRIALS):

        self.model = model

        self.direction = direction
        self.metric = metric
        self.n_trials = n_trials
        self.best_params = {}
        self.study = None

    def tuning(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame,
               y_valid: pd.Series, save_mode=False, resume: bool=True):
        # Подбор записывается в файл на диске (config.OPTUNA_DB_FILE): Optuna сама
        # дописывает туда каждую попытку сразу после её завершения.
        #   resume=True  — продолжить прошлый подбор этой модели, если он есть;
        #   resume=False — стереть прошлый подбор и начать с нуля. Так нужно делать,
        #                  когда поменялись данные, признаки или границы гиперпараметров:
        #                  старые попытки с новыми сравнивать уже нельзя.
        config.OPTUNA_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        storage = f'sqlite:///{config.OPTUNA_DB_FILE.as_posix()}'
        study_name = f'find_hp_{_model_type(self.model)}'

        if not resume:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except KeyError:
                pass  # подбора с таким именем ещё не было, удалять нечего

        # load_if_exists=True: если подбор с таким именем уже лежит в файле,
        # он открывается вместе со всеми прошлыми попытками, а не создаётся заново.
        study = optuna.create_study(direction=self.direction, study_name=study_name,
                                    storage=storage, load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))

        # Считаем только завершённые попытки: та, на которой подбор оборвался,
        # осталась в файле незаконченной и в зачёт не идёт.
        n_done = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
        n_left = max(self.n_trials - n_done, 0)
        print(f'Подбор {study_name}: готово попыток {n_done}, осталось {n_left}')

        # Первые 10 попыток Optuna выбирает случайно. С одним и тем же зерном генератор
        # после перезапуска выдал бы те же самые точки, что и в прошлый раз, и попытки
        # ушли бы на повтор уже проверенного. Поэтому при продолжении зерно сдвигается
        # на число готовых попыток. При запуске с нуля n_done = 0, и зерно прежнее.
        study.sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED + n_done)
        objective: Callable | None=None

        def get_score(model: Model=None):
            return self.metric(y_true=y_valid, y_pred=model.predict(x_valid))


        if self.model is BaseModel:
            def objective(trial):
                v = trial.suggest_float('config_value', 1, 5)
                c = trial.suggest_categorical('strategy', ['mean', 'constant'])
                model = Model(model=BaseModel, strategy=c, config_value=v).fit(x_train, y_train)
                return get_score(model)

        elif self.model is CatBoostRegressor:
            def objective(trial):
                depth = trial.suggest_int('depth', 4, 10)

                learning_rate = trial.suggest_float('learning_rate', 0.03, 0.3, log=True)
                l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True)

                model = Model(model=CatBoostRegressor, iterations=config.CATBOOST_MAX_ITERATIONS,
                                        depth=depth,
                                        learning_rate=learning_rate,
                                        l2_leaf_reg=l2_leaf_reg, verbose=False
                                       ).fit(x_train, y_train,
                                            eval_set=(x_valid, y_valid),
                                            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)

                trial.set_user_attr('iterations', model.model.best_iteration_ + 1)
                return get_score(model)

        elif self.model is LGBMRegressor:
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 50, 1000)
                max_depth = trial.suggest_int('max_depth', 3, 12)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                num_leaves = trial.suggest_int('num_leaves', 5, 50)
                reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10, log=True)
                model = Model(model=LGBMRegressor, n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        num_leaves=num_leaves,
                                        reg_lambda=reg_lambda, verbose=-1).fit(x_train, y_train)
                return get_score(model)

        elif self.model is XGBRegressor:
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 50, 1000)
                max_depth = trial.suggest_int('max_depth', 3, 12)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                max_leaves = trial.suggest_int('max_leaves', 5, 50)
                model = Model(model=XGBRegressor, n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        max_leaves=max_leaves, verbosity=0).fit(x_train, y_train)
                return get_score(model)

        # Запускаем только недостающие попытки. Если все n_trials уже сделаны,
        # новых не будет, и ниже просто возьмутся лучшие параметры из файла.
        if n_left > 0:
            study.optimize(objective, n_trials=n_left)


        self.best_params = study.best_params | study.best_trial.user_attrs
        self.study = study

        if save_mode:
            path_params = config.PARAMS_FILES[_model_type(self.model)]
            path_params.parent.mkdir(parents=True, exist_ok=True)

            with open(path_params, 'w', encoding='utf-8') as f:
                json.dump(self.best_params, f, indent=2)

        return self
    



class Model():
    def __init__(self, model: type[BaseModel]
                              |type[CatBoostRegressor]
                              |type[LGBMRegressor]
                              |type[XGBRegressor], **params):
        self.params = params
        self.metrics = None
        self.model_type = _model_type(model)
        self.model = model(**params)


    def load_model(self, path: Path=None):
        if path is None:
            path = config.MODEL_FILES[self.model_type]

        if not Path(path).exists():
            raise FileNotFoundError(f'Нет сохранённой модели: {path}')

        if self.model_type == 'base':
            self.model = joblib.load(path)
        elif self.model_type in ['catboost', 'xgb'] :
            self.model.load_model(str(path))
        elif self.model_type == 'lgbm':
            self.model = lightgbm.Booster(model_file=str(path))
        return self

    def fit(self, x: pd.DataFrame, y: pd.Series, save_mode: bool=False, **fit_params):
        self.model.fit(x, y, **fit_params)

        if save_mode and self.model_type is not None:
            path_model = config.MODEL_FILES[self.model_type]
            path_model.parent.mkdir(parents=True, exist_ok=True)

            if self.model_type == 'base':
                joblib.dump(self.model, path_model)

            if self.model_type == 'catboost':
                self.model.save_model(str(path_model))

            if self.model_type == 'xgb':
                self.model.save_model(str(path_model))

            if self.model_type == 'lgbm':
                self.model.booster_.save_model(str(path_model))



        return self

    def predict(self, x: pd.DataFrame):
        return self.model.predict(x)


    def get_metrics(self, x: pd.DataFrame, y: pd.Series, save_mode: bool=False):
        y_score = self.model.predict(x)
        metrics = {
            'MSE': MSE(y_true=y, y_pred=y_score),
            'MAE': MAE(y_true=y, y_pred=y_score),
            'MAPE': MAPE(y_true=y, y_pred=y_score)
        }
        self.metrics = metrics

        if save_mode:
            path_metrics = config.METRICS_FILES.get(self.model_type, config.OTHER_METRICS_FILE)
            path_metrics.parent.mkdir(parents=True, exist_ok=True)

            with open(path_metrics, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)

        return metrics
