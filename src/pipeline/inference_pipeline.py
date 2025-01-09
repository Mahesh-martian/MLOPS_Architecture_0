import sys
import os
from src.utils.utils import fetch_path_from_config
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import load_object
from src.constants.constants import CONFIG_PATH
import pandas as pd
import requests
import warnings

warnings.filterwarnings('ignore')



class InferencePipeline:
    def __init__(self):
        pass

    def get_data(self, api):

        response = requests.get(api)
        res = response.json()
        df = pd.DataFrame(data = res, index=[1])
        inference_data = df.drop(['Activity','date_time'], axis=1)
        ground_truth = df['Activity']

        return inference_data, ground_truth

    def predict(self,features):
        try:
            model_register_path = fetch_path_from_config("Paths", "model_registery_path", CONFIG_PATH)
            preprocessor_path=os.path.join(model_register_path,'artifacts','pretrained_weights')
            preprocessor=load_object(preprocessor_path, 'preprocessor.joblib')
            feature_path = os.path.join(model_register_path,'best_features')
            best_features = load_object(feature_path, 'top_10_features.joblib').tolist()
            model_path=os.path.join(model_register_path,'best_model')
            model=load_object(model_path, 'best_model.joblib')
            pred=model.predict(features[best_features])
            return pred
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

if __name__ == '__main__':

    Infer_pipeline = InferencePipeline()
    features , ground_truth = Infer_pipeline.get_data(api="http://127.0.0.1:8888/latest")
    prediction = Infer_pipeline.predict(features)
    if prediction[0] == ground_truth.values[0]:
        print("correct prediction")
    else:
        print('Incorrect')

    print("prediction --> ",prediction, "ground_truth -->",ground_truth)



