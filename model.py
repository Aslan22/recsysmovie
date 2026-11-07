from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import optuna
import json
from typing import Callable
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error as RMSE, mean_absolute_error as MAE, mean_squared_error as MSE
import joblib
import pathlib

class BaseModel():
    def __init__(self, strategy: str='mean', config_value: float=3.0):
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


class Tuner():
    def __init__(self, model: type[BaseModel]
                              |type[CatBoostRegressor]
                              |type[LGBMRegressor]
                              |type[XGBRegressor], direction: str='minimize',
                 metric: Callable=MSE, n_trials: int=50,
                 load_path: str=None):
        if load_path is None:
            self.model = model
        else:
            self.model = None

        self.direction = direction
        self.metric = metric
        self.n_trials = n_trials
        self.best_params = {}
        self.study = None

    def tuning(self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame,
               y_valid: pd.Series, save_mode=False):
        study = optuna.create_study(direction=self.direction, study_name='find_hp',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        objective: Callable | None=None

        def get_score(model: Model=None):
            model.fit(x_train, y_train)
            return self.metric(y_true=y_valid, y_pred=model.predict(x_valid))


        if self.model is BaseModel:
            def objective(trial):
                v = trial.suggest_float('config_value', 1, 5)
                c = trial.suggest_categorical('strategy', ['mean', 'constant'])
                return get_score(Model(model=BaseModel, strategy=c, config_value=v))

        elif self.model is CatBoostRegressor:
            def objective(trial):
                iterations = trial.suggest_int('iterations', 50, 1000)
                depth = trial.suggest_int('depth', 4, 10)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True)
                return get_score(Model(model=CatBoostRegressor, iterations=iterations,
                                        depth=depth,
                                        learning_rate=learning_rate,
                                        l2_leaf_reg=l2_leaf_reg, verbose=False))

        elif self.model is LGBMRegressor:
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 50, 1000)
                max_depth = trial.suggest_int('max_depth', 3, 12)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                num_leaves = trial.suggest_int('num_leaves', 5, 50)
                reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10, log=True)
                return get_score(Model(model=LGBMRegressor, n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        num_leaves=num_leaves,
                                        reg_lambda=reg_lambda, verbose=-1))

        elif self.model is XGBRegressor:
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 50, 1000)
                max_depth = trial.suggest_int('max_depth', 3, 12)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                max_leaves = trial.suggest_int('max_leaves', 5, 50)
                return get_score(Model(model=XGBRegressor, n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        max_leaves=max_leaves, verbosity=0))

        study.optimize(objective, n_trials=self.n_trials)

        self.best_params = study.best_params
        self.study = study

        if save_mode:
            path_params = 'C:/Users/aslan/PycharmProjects/recsysmovie/artifacts/models'
            if self.model is BaseModel:
                path_params += '/base_model/base_params.json'

            if self.model is CatBoostRegressor:
                path_params += '/catboost/catboost_params.json'

            if self.model is XGBRegressor:
                path_params += '/xgboost/xgboost_params.json'

            if self.model is LGBMRegressor:
                path_params += '/lightgbm/lightgbm_params.json'

            with open(path_params, 'w', encoding='utf-8') as f:
                json.dump(self.best_params, f, indent=2)

        return self
    



class Model():
    def __init__(self, model: type[BaseModel]
                              |type[CatBoostRegressor]
                              |type[LGBMRegressor]
                              |type[XGBRegressor], **params):
        self.params = params
        self.model_type = None
        self.metrics = None
        if model is BaseModel:
            self.model_type = 'base'
        elif model is CatBoostRegressor:
            self.model_type = 'catboost'
        elif model is LGBMRegressor:
            self.model_type = 'lgbm'
        elif model is XGBRegressor:
            self.model_type = 'xgb'

        self.model = model(**params)

    def fit(self, x: pd.DataFrame, y: pd.Series, save_mode: bool=False):
        self.model.fit(x, y)

        if save_mode:
            path_model = 'C:/Users/aslan/PycharmProjects/recsysmovie/artifacts/models'
            if self.model_type == 'base':
                path_model += '/base_model/base.joblib'
                joblib.dump(self.model, path_model)

            if self.model_type == 'catboost':
                path_model += '/catboost/catboost.cbm'
                self.model.save_model(path_model)

            if self.model_type == 'xgb':
                path_model += '/xgboost/xgboost.json'
                self.model.save_model(path_model)

            if self.model_type == 'lgbm':
                path_model += '/lightgbm/lightgbm.txt'
                self.model.booster_.save_model(path_model)



        return self

    def predict(self, x: pd.DataFrame):
        return self.model.predict(x)


    def get_metrics(self, x: pd.DataFrame, y: pd.Series, save_mode: bool=False):
        y_score = self.model.predict(x)
        metrics = {
            'RMSE': RMSE(y_true=y, y_pred=y_score),
            'MSE': MSE(y_true=y, y_pred=y_score),
            'MAE': MAE(y_true=y, y_pred=y_score),
        }
        self.metrics = metrics

        if save_mode:
            path_metrics = 'C:/Users/aslan/PycharmProjects/recsysmovie/reports'
            if self.model_type == 'base':
                path_metrics += '/basemodel_metrics.json'
            elif self.model_type == 'catboost':
                path_metrics += '/catboost_metrics.json'
            elif self.model_type == 'lgbm':
                path_metrics += '/lgbm_metrics.json'
            elif self.model_type == 'xgb':
                path_metrics += '/xgb_metrics.json'
            else:
                path_metrics += '/other_metrics.json'

            with open(path_metrics, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)

        return metrics
