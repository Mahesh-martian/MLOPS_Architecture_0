import sys
import os
from pathlib import Path
from dataclasses import dataclass
from src.logger.logging import logging
from src.exception.exception import CustomException
from src.components.data_ingestion import DataIngestionconfig
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from src.utils.utils import save_object, load_object, fetch_path_from_config
from src.constants.constants import BASE_DIR

@dataclass
class DataTransformationConfig:
    CONFIG_PATH = os.path.join(BASE_DIR,"config.ini")
    model_registry_path = fetch_path_from_config("Paths", "model_registery_path", CONFIG_PATH)
    preprocessor_obj_file_path=os.path.join(model_registry_path,r'artifacts\pretrained_weights')
    best_features_path = os.path.join(model_registry_path, 'best_features')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Activity']
            logging.info('Pipeline Initiated')

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('labelencoder',LabelEncoder)
                ]
            )
            preprocessor=ColumnTransformer([
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        else:
            logging.info('Pipeline Completed')

    def initaite_data_transformation(self,train_path,test_path):
            try:
                # Reading train and test data
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info('Read train and test data completed')

                logging.info('Obtaining preprocessing object')

                preprocessing_obj = self.get_data_transformation_object()

                target_column_name = 'Activity'
                drop_columns = [target_column_name,'date_time']

                logging.info('fetching best features')
                best_features = load_object(file_path=self.data_transformation_config.best_features_path, filename='top_10_features.joblib')
                logging.info('fetched best features successfully')

                input_feature_train_df = train_df[best_features]
                target_feature_train_df=train_df[target_column_name]

                input_feature_test_df=test_df[best_features]
                target_feature_test_df=test_df[target_column_name]

                logging.info(f'Train Dataframe Head : \n{input_feature_train_df.head().to_string()}')
                logging.info(f'Test Dataframe Head  : \n{input_feature_test_df.head().to_string()}')
                
                ## Trnasformating using preprocessor obj
                input_feature_train_arr=input_feature_train_df.to_numpy()
                input_feature_test_arr=input_feature_test_df.to_numpy()

                logging.info("Applying preprocessing object on training and testing datasets.")
                

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                save_object(

                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj,
                    file_name = '/preprocessor.joblib'

                )
                logging.info('Preprocessor joblib file saved')

                logging.info('saving training data and testing data')

                train_dvc = os.path.join(self.data_transformation_config.model_registry_path, 'data_versions')
                save_object(file_path= train_dvc, obj = train_arr ,  file_name='/train_array.joblib', )
                test_dvc = os.path.join(self.data_transformation_config.model_registry_path, 'data_versions')
                save_object(file_path= test_dvc, obj = test_arr ,  file_name='/test_array.joblib', )

                logging.info('training data and testing data in data_versioning folder')
                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )
                
            except Exception as e:
                logging.info("Exception occured in the initiate_datatransformation")

                raise CustomException(e,sys)
            
if __name__ == "__main__":
    train_path = DataIngestionconfig.train_data_path
    test_path = DataIngestionconfig.test_data_path
    data_trans_obj = DataTransformation()
    data_trans_obj.initaite_data_transformation(train_path, test_path)
