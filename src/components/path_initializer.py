import os
import configparser
from src.utils.utils import create_model_registery, add_paths_to_config,add_path_to_config, fetch_path_from_config
from pathlib import Path
from src.constants.constants import BASE_DIR, CONFIG_PATH
from src.logger.logging import logging
import dataclasses
# CONFIG_PATH = os.path.join(BASE_DIR,"config.ini")

class PathInitializer:

    def add_all_paths_to_config(self):

        self.registery_path = os.path.join(BASE_DIR, "model_registery")
        self.data_path = os.path.join(BASE_DIR, "data/raw_sample.csv")

        logging.info("Adding paths to config.ini file")
        self.all_paths = {
            "base_dir": BASE_DIR, 
            "registery_path": self.registery_path,
            "data_path": self.data_path}

        add_paths_to_config(self.all_paths, CONFIG_PATH)
        logging.info("Added paths successfully")

    def initialize_model_registery(self):

        logging.info("creating model register")
        model_registery_path = create_model_registery()
        add_path_to_config('model_registery_path',model_registery_path , CONFIG_PATH, "Paths")
        new_path = fetch_path_from_config("Paths","model_registery_path" , CONFIG_PATH)        
        logging.info("created model registery: {}".format(new_path))

        return new_path


Initialize_paths = PathInitializer()
Initialize_paths.add_all_paths_to_config()
Initialize_paths.initialize_model_registery()