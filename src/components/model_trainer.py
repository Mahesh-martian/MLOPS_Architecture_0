import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import save_object,fetch_path_from_config, load_object
from src.utils.utils import evaluate_model

from dataclasses import dataclass
import sys
import os
from src.constants.constants import BASE_DIR,CONFIG_PATH

# CONFIG_PATH = os.path.join(BASE_DIR,"config.ini")


@dataclass 
class ModelTrainerConfig:
    model_registry_path = fetch_path_from_config("Paths", "model_registery_path", CONFIG_PATH)
    par_dir = BASE_DIR
    trained_model_file_path = os.path.join(model_registry_path, 'best_model')
    train_array_file_path = os.path.join(model_registry_path, 'data_versions')
    test_array_file_path = os.path.join(model_registry_path, 'data_versions')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array_path,test_array_path):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            train_array = load_object(train_array_path, 'train_array.joblib')
            test_array = load_object(test_array_path,'test_array.joblib')
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LogisticRegression':LogisticRegression(max_iter=1000),
            'DecisionTreeClassifier':DecisionTreeClassifier(random_state=42),
            'RandomForestClassifier':RandomForestClassifier(n_estimators=100, random_state=60),
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
                file_name= '/best_model.joblib'
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

if __name__ == '__main__':
    train_array_path = ModelTrainerConfig.train_array_file_path
    test_array_path = ModelTrainerConfig.test_array_file_path

    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_array_path, test_array_path)