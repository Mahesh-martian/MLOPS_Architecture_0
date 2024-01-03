import os
import pandas as pd
from pathlib import Path
from src.utils.utils import create_model_registery
from src.exception.exception import CustomException
from src.utils.utils import move_to_parent_dir, save_object, get_top_K_features, load_object
from sklearn.ensemble import ExtraTreesClassifier

# print(registry_path)

# res = move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0', )
# print(res)
# dir = current_dir = os.path.dirname(os.path.abspath(__file__))
# print(dir.split('\\')[-1])


# Parent_dir = move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0')
# data_path = os.path.join(Parent_dir, 'Data', 'Raw_data.gzip')
# df = pd.read_parquet(data_path)
# print(df.head())


# train_data_path:str = os.path.join(Parent_dir, 'artifacts', 'train.csv')
# test_data_path:str = os.path.join(Parent_dir, 'artifacts', 'test.csv')
# raw_data_path:str = os.path.join(Parent_dir, 'artifacts', 'raw.csv')

# print(train_data_path)
# print(test_data_path)
# print(raw_data_path)
# The line `Parent_dir = move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0')` is calling the
# `move_to_parent_dir` function and assigning the returned value to the variable `Parent_dir`.




# class Demo:
#     pass

# d1 = Demo()
# model_registry_path = create_model_registery(folder_name = 'model_registery')
# print(model_registry_path)
# preprocessor_obj_file_path=os.path.join(model_registry_path,r'artifacts\weights\preprocessor.joblib')
# # preprocessor_obj_file_path = preprocessor_obj_file_path.replace("\\" , "/")
# save_object(preprocessor_obj_file_path, d1, '/preprocessor.joblib')

# res = preprocessor_obj_file_path.split('\\')[:-1]
# res = "/".join(res)

# if os.path.exists(Path(res)):
#     print('True')
# else:
#     print('False')


# Parent_dir = move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0')
# model_registry_path = create_model_registery(folder_name = 'model_registery')
# print(model_registry_path)


# par_dir = move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0')
# train_path = os.path.join(par_dir, 'artifacts\\train.csv')
# print(train_path)
# df = pd.read_csv(train_path)
# X = df.drop(['Activity','date_time'] , axis=1)
# y = df['Activity']

# top_12_features = get_top_K_features(kvalue=10, tree_clf = ExtraTreesClassifier, X = X, Y = y)
# print(top_12_features)
# model_registry_path = create_model_registery(folder_name = 'model_registery')
# print(model_registry_path)
# best_features_path = os.path.join(model_registry_path, 'best_features')
# print(best_features_path)
# def select_best_features(X, y):

#         df = pd.read_csv(train_path)
#         X = df.drop(['Activity','date_time'] , axis=1)
#         y = df['Activity']
#         top_10_features = get_top_K_features(kvalue=10, tree_clf = ExtraTreesClassifier, X = X, Y = y)
#         save_object(file_path =best_features_path , obj = top_10_features , file_name = '/top_10_features.joblib')
#         top_12_features = get_top_K_features(kvalue=12, tree_clf = ExtraTreesClassifier, X = X, Y = y)
#         save_object(file_path =best_features_path , obj = top_12_features , file_name = '/top_12_features.joblib')

# select_best_features(X, y)


# top_12_features = load_object(file_path=best_features_path, filename='top_12_features.joblib')
# print(top_12_features)

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# models={
# 'LogisticRegression':LogisticRegression(max_iter=1000),
# 'DecisionTreeClassifier':DecisionTreeClassifier(random_state=42),
# 'RandomForestClassifier':RandomForestClassifier(n_estimators=100, random_state=60),
# }

# report  = {}

# for model in models:
#     res = models[model]
#     print(model)
#     print(type(model))

#     report[model] = 'train'

# print(report)

# from src.components.model_trainer import *


# train_array_path = ModelTrainerConfig.train_array_file_path
# test_array_path = ModelTrainerConfig.test_array_file_path


# res = load_object(test_array_path,'test_array.joblib')
# print(type(res))



import os
from pathlib import Path
import sys

def create_model_registry(base_folder='model_registery'):
    try:
        parent_dir = move_to_parent_dir(parent_folder='BlackMi_mobiles_v_0')
        base_dir = os.path.join(parent_dir, base_folder)

        # Find existing versions
        existing_versions = [d for d in os.listdir(base_dir) if d.startswith(f'{base_folder}_ver_') and os.path.isdir(os.path.join(base_dir, d))]

        # Determine the next version
        if existing_versions:
            latest_version = max(int(ver.split('_')[-1]) for ver in existing_versions)
            next_version = latest_version + 1
        else:
            next_version = 1

        # Create the new versioned directory
        new_version_dir = f'{base_folder}_ver_{next_version:02d}'
        dir_path = os.path.join(base_dir, new_version_dir)

        os.makedirs(dir_path, exist_ok=True)
        return Path(dir_path)

    except Exception as e:
        # Handle exceptions here
        raise CustomException(e, sys)

# Example usage
new_model_registry = create_model_registry()
print(f"Created model registry directory: {new_model_registry}")
