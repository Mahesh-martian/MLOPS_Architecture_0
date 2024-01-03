import os
import sys
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from src.exception.exception import CustomException
from src.logger.logging import logging

def move_to_parent_dir(parent_folder):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up the directory tree
    parent_dir = current_dir

    while True:

        if parent_dir.split('\\')[-1] == parent_folder:
            break
        else:
            parent_dir = os.path.dirname(parent_dir)
    return Path(parent_dir)


    # Construct the absolute path to the Raw_data.gzip file
    data_path = os.path.join(project_dir)
    return Path(data_path)


def create_model_registery(folder_name):
    try:
        parent_dir =  move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0')
        dir_path = os.path.join(parent_dir, folder_name)
        if os.path.exists(dir_path):
            return Path(dir_path)
        else:
            os.makedirs(dir_path, exist_ok=True)
            return Path(dir_path)

    except Exception as e:
        raise CustomException(e,sys)


def save_object(file_path, obj, file_name):
    try:
        dir = "/".join(file_path.split('\\'))
        if os.path.exists(Path(dir)):
            joblib.dump(obj, dir+file_name)
        else:
            os.makedirs(dir, exist_ok=True)
            joblib.dump(obj, dir+file_name)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path,filename):
    try:
        dir = os.path.join(file_path, filename)
        file = "/".join(dir.split('\\'))

        return joblib.load(file)
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    

def get_top_K_features(kvalue, tree_clf, X, Y):

    clf = tree_clf(n_estimators=150)
    clf = clf.fit(X,Y)
    feature_df = pd.DataFrame(data=(X.columns, clf.feature_importances_)).T.sort_values(by=1, ascending=False)
    cols = feature_df.head(kvalue)[0].values
    return cols



def save_best_features(file_path, kvalue, X):

    try:
        file_path = Path(file_path+str(kvalue)+'features.joblib' )
        K_feature = np.array(X.columns)
        joblib.dump(K_feature, file_path)

    except Exception as e:
        raise CustomException(e, sys)
    

def save_datasets(x, file_path, kvalue):
    try:
        file_path = Path(file_path+str(kvalue)+'_dataset.joblib' )
        X_dataset = np.array(x)
        joblib.dump(X_dataset, file_path)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train,y_train,X_test,y_test,models):
    
    try:
        Report = {}

        for mod in models:
            model = models[mod]

            model.fit(X_train,y_train)
            train_accuracy = round(model.score(X_train, y_train)*100,2)
            test_accuracy = round(model.score(X_test, y_test)*100,2)

            Report[(mod)] = test_accuracy

        return Report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

        


