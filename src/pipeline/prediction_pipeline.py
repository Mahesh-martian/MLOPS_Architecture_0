import sys
import os
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import load_object, create_model_registery
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_register_path = create_model_registery()
            preprocessor_path=os.path.join(model_register_path,'pretrained_weights')
            model_path=os.path.join(model_register_path,'best_model')

            preprocessor=load_object(preprocessor_path, 'preprocessor.joblib')
            model=load_object(model_path, 'best_model.joblib')

            data_scaled=preprocessor.inverse_transform(features)

            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)