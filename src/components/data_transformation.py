import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.components.data_ingestion import DataIngestionConfig

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('D:/StudentPerformance_ML/src/artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.data_ingestion_config = DataIngestionConfig()

    def read_data(self, file_path: str):
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_data_transformer_object(self, numerical_cols, categorical_cols, transform_cols):
        logging.info("Entered the data transformation")

        try:
            numeric_transformer = StandardScaler()
            one_hot_transformer = OneHotEncoder()
            power_transform = Pipeline(steps= [('transformer', PowerTransformer(method='yeo-johnson'))])

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder",one_hot_transformer,categorical_cols)
                    ("StandardScaler",numeric_transformer,numerical_cols)
                    ("Transformer",power_transform,transform_cols)
                ]
            )

            logging.info("Created data preprocessor object using Column Transformer")

            logging.info("Exited the data transformation function")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def initiate_data_transformation(self):
        try:

            preprocessor = self.get_data_transformer_object()

            train_df =self.read_data(self.data_ingestion_config.train_data_path)
            test_df = self.read_data(self.data_ingestion_config.test_data_path)

            target_column_name="math_score"

            X_train = train_df.drop(target_column_name, axis = 1)
            Y_train = train_df[target_column_name]

            X_test = test_df.drop(target_column_name, axis = 1)
            Y_test = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.fit_transform(X_test)

            train_arr = np.c_[
                X_train_arr, np.array(Y_train)
            ]
            test_arr = np.c_[X_test_arr, np.array(Y_test)]

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            logging.info(f"Saved preprocessing object at:{self.transformation_config.preprocessor_obj_file_path}") 
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys) from e




            

