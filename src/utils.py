import os
import sys
import dill
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException


def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        #wb - write byte mode
        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
            raise CustomException(e, sys) from e


def evaludate_models(X_train, Y_train, X_test, Y_test, models) -> dict:
    try:
        report = {}

        for model_name, model in models.items():
             
             model.fit(X_train, Y_train) #train the model

             Y_train_pred = model.predict(X_train)
             Y_test_pred = model.predict(Y_train)

             train_model_score = r2_score(Y_train, Y_train_pred)
             test_model_score = r2_score(Y_test, Y_test_pred)

             report[model_name]=test_model_score

        return report

    except Exception as e:
            raise CustomException(e, sys) from e