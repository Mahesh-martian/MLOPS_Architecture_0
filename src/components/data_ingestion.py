import os
import sys
from src.logger.logging import logging
from src.exception.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from dataclasses import dataclass
# from src.components.data_transformation import *
from pathlib import Path
from src.utils.utils import move_to_parent_dir, get_top_K_features, save_object,add_path_to_config, fetch_path_from_config
import configparser
## Intitialize the Data Ingetion Configuration
from src.constants.constants import BASE_DIR,CONFIG_PATH

# CONFIG_PATH = os.path.join(BASE_DIR,"config.ini")


@dataclass
class DataIngestionconfig:
    Parent_dir = fetch_path_from_config("Paths", "model_registery_path", CONFIG_PATH)
    logging.info("current model registery path: {}".format(Parent_dir))
    data_path = fetch_path_from_config("Paths", "data_path", CONFIG_PATH)
    train_data_path:str = os.path.join(Parent_dir, 'artifacts', 'train.csv')
    test_data_path:str = os.path.join(Parent_dir, 'artifacts', 'test.csv')
    raw_data_path:str = os.path.join(Parent_dir, 'artifacts', 'raw.csv')
    best_features_path = os.path.join(Parent_dir, 'best_features')


class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionconfig()

    def select_best_features(self):
        logging.info("selecting best features from raw data")
        df = pd.read_csv(self.ingestion_config.data_path)
        X = df.drop(['Activity','date_time'] , axis=1)
        y = df['Activity']
        top_10_features = get_top_K_features(kvalue=10, tree_clf = ExtraTreesClassifier, X = X, Y = y)
        save_object(file_path = self.ingestion_config.best_features_path , obj = top_10_features , file_name = '/top_10_features.joblib')
        top_12_features = get_top_K_features(kvalue=12, tree_clf = ExtraTreesClassifier, X = X, Y = y)
        save_object(file_path = self.ingestion_config.best_features_path , obj = top_12_features , file_name = '/top_12_features.joblib')
        logging.info(f"added best features to folder{self.ingestion_config.best_features_path}")

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            data_path = os.path.join(BASE_DIR, 'Data', 'Raw_data.csv')
            df=pd.read_csv(data_path)
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)


if __name__ == "__main__":
    data = DataIngestion()
    data.select_best_features()
    data.initiate_data_ingestion()

