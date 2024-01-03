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
from src.utils.utils import move_to_parent_dir, get_top_K_features, save_object, create_model_registery

## Intitialize the Data Ingetion Configuration

model_registry_path = create_model_registery(folder_name = 'model_registery')
Parent_dir = move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0')


@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join(Parent_dir, 'artifacts', 'train.csv')
    test_data_path:str = os.path.join(Parent_dir, 'artifacts', 'test.csv')
    raw_data_path:str = os.path.join(Parent_dir, 'artifacts', 'raw.csv')
    best_features_path = os.path.join(model_registry_path, 'best_features')


class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionconfig()

    def select_best_features(self):
        
        df = pd.read_csv(self.ingestion_config.train_data_path)
        X = df.drop(['Activity','date_time'] , axis=1)
        y = df['Activity']
        top_10_features = get_top_K_features(kvalue=10, tree_clf = ExtraTreesClassifier, X = X, Y = y)
        save_object(file_path = self.ingestion_config.best_features_path , obj = top_10_features , file_name = '/top_10_features.joblib')
        top_12_features = get_top_K_features(kvalue=12, tree_clf = ExtraTreesClassifier, X = X, Y = y)
        save_object(file_path = self.ingestion_config.best_features_path , obj = top_12_features , file_name = '/top_12_features.joblib')

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            data_path = os.path.join(Parent_dir, 'Data', 'Raw_data.csv')
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

