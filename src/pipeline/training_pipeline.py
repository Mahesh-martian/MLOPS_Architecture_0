import time
import os
import sys
from src.logger.logging import logging
from src.exception.exception import CustomException
import pandas as pd
from src.components import path_initializer
from src.components.data_ingestion import DataIngestion, DataIngestionconfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


if __name__ == '__main__':

    obj = DataIngestion()
    obj.select_best_features()
    obj.initiate_data_ingestion()
    train_path = DataIngestionconfig.train_data_path
    test_path = DataIngestionconfig.test_data_path
    data_trans_obj = DataTransformation()
    data_trans_obj.initaite_data_transformation(train_path, test_path)
    train_array_path = ModelTrainerConfig.train_array_file_path
    test_array_path = ModelTrainerConfig.test_array_file_path
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_array_path, test_array_path)
