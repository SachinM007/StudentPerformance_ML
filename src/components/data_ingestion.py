#read or get data, perform the train test spplit and store the data

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('D:/StudentPerformance_ML/src/artifacts','train.csv')
    test_data_path: str = os.path.join('D:/StudentPerformance_ML/src/artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component and initiated")
        try:
            df = pd.read_csv(r"data\StudentsPerformance.csv")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            train_data, test_data = train_test_split(df, test_size=0.2,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header =True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header =True)

            logging.info("Ingestion of data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys) from e