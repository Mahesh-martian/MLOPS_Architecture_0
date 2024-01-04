import os
import configparser
from src.utils.utils import create_model_registery, update_model_register_path, add_paths_to_config, move_to_parent_dir,add_path_to_config
from pathlib import Path
from src.components import CONFIG_PATH, BASE_DIR
from src.logger.logging import logging



class PathInitializer:

    def __init__(self):
        pass

    def add_all_paths_to_config(self):

        self.registery_path = os.path.join(BASE_DIR, "model_registery")
        self.data_path = os.path.join(BASE_DIR, "data/raw_sample.csv")

        self.all_paths = {
            "base_dir": BASE_DIR, 
            "registery_path": self.registery_path,
            "data_path": self.data_path}

        add_paths_to_config(self.all_paths, CONFIG_PATH)

    def initialize_model_registery(self):

        model_registery_path = create_model_registery()

        add_path_to_config('model_registery_path',model_registery_path , CONFIG_PATH, "Paths")



if __name__ == "__main__":
    Initialize_paths = PathInitializer()
    Initialize_paths.add_all_paths_to_config()
    Initialize_paths.initialize_model_registery()