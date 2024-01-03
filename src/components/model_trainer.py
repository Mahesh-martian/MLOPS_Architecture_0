import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import save_object, create_model_registery,get_top_K_features, move_to_parent_dir, load_object
from src.utils.utils import evaluate_model

from dataclasses import dataclass
import sys
import os
from pathlib import Path
import joblib

model_registry_path = create_model_registery(folder_name = 'model_registery')
par_dir = move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0')

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = Path(os.path.join(model_registry_path, 'pretrained_model','model.pkl'))
    train_array_file_path = os.path.join(model_registry_path, 'data_versions','train_array.joblib')
    test_array_file_path = os.path.join(model_registry_path, 'data_versions','test_array.joblib')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
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
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)


# Save the training and test data in model registery folder to maintain data version control for feature purpose
# load the file from the model registery folder to model trainer the train the model 