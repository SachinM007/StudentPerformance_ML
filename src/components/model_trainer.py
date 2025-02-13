import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaludate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('D:/StudentPerformance_ML/src/artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_obj):
        try:
            logging.info("Splitting into train and test data")

            X_train, Y_train, X_test, Y_test = (
                train_arr[:,:,-1], #all rows & all cols except last column
                train_arr[:,-1], # all rows of last column
                test_arr[:,:,-1],
                test_arr[:,-1]
            )

            models = {
                "Linear regressor": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor,
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor(),
            }

            model_report: dict = evaludate_models(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test = Y_test, models=models)
        
        
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found, check the data passed and models used")

            logging.info("model evaluation has been completed")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)

            return r2_score(Y_test, predicted)


        except Exception as e:
            raise CustomException(e, sys) from e
        
