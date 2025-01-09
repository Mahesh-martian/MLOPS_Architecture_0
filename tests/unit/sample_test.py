import pytest
import os
import joblib
import pandas as pd
import numpy as np
from unittest.mock import patch
from pathlib import Path
import configparser
from src.utils.utils import (
    move_to_parent_dir, create_model_registery, save_object, load_object,
    get_top_K_features, save_best_features, save_datasets, evaluate_model,
    update_model_register_path, add_paths_to_config, add_path_to_config,
    fetch_path_from_config
)

def test_move_to_parent_dir():
    # Test navigating to a known parent folder
    parent_folder = "known_parent_folder"
    with patch("os.path.dirname", side_effect=["/test/child", "/test", "/"]):
        assert move_to_parent_dir(parent_folder) == parent_folder


def test_create_model_registery():
    base_folder = "test_model_registry"
    parent_folder = "test_parent_folder"
    with patch("os.makedirs"):
        with patch("os.listdir", return_value=[]):
            result = create_model_registery(base_folder)
            assert base_folder in result


def test_save_object():
    file_path = "test_dir"
    file_name = "test_file.joblib"
    obj = {"key": "value"}

    save_object(file_path, obj, file_name)
    assert os.path.exists(Path(file_path, file_name))


def test_load_object():
    file_path = "test_dir"
    file_name = "test_file.joblib"
    obj = {"key": "value"}

    save_object(file_path, obj, file_name)
    loaded_obj = load_object(file_path, file_name)
    assert loaded_obj == obj


def test_get_top_K_features():
    from sklearn.ensemble import RandomForestClassifier

    X = pd.DataFrame(np.random.rand(10, 5), columns=[f"feature_{i}" for i in range(5)])
    Y = np.random.randint(0, 2, 10)

    top_features = get_top_K_features(3, RandomForestClassifier, X, Y)
    assert len(top_features) == 3


def test_save_best_features():
    file_path = "test_features_dir/"
    kvalue = 5
    X = pd.DataFrame(np.random.rand(10, kvalue), columns=[f"feature_{i}" for i in range(kvalue)])

    save_best_features(file_path, kvalue, X)
    saved_file = Path(file_path + str(kvalue) + "features.joblib")
    assert saved_file.exists()


def test_save_datasets():
    file_path = "test_datasets_dir/"
    kvalue = 5
    x = pd.DataFrame(np.random.rand(10, kvalue))

    save_datasets(x, file_path, kvalue)
    saved_file = Path(file_path + str(kvalue) + "_dataset.joblib")
    assert saved_file.exists()


def test_evaluate_model():
    from sklearn.ensemble import RandomForestClassifier

    X_train = np.random.rand(50, 5)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)

    models = {"RandomForest": RandomForestClassifier()}
    report = evaluate_model(X_train, y_train, X_test, y_test, models)
    assert "RandomForest" in report


def test_update_model_register_path():
    config_file = "test_config.ini"
    new_paths = {"new_model_registry": "test/path/to/registry"}

    config = configparser.ConfigParser()
    config["Paths"] = {}
    with open(config_file, "w") as f:
        config.write(f)

    updated_path = update_model_register_path(new_paths, config_file)
    assert updated_path == new_paths["new_model_registry"]


def test_add_paths_to_config():
    config_file = "test_config.ini"
    new_paths = {"test_path": "test/path"}

    config = configparser.ConfigParser()
    config["Paths"] = {}
    with open(config_file, "w") as f:
        config.write(f)

    add_paths_to_config(new_paths, config_file)
    config.read(config_file)
    assert config["Paths"]["test_path"] == "test/path"


def test_add_path_to_config():
    config_file = "test_config.ini"
    path_name = "test_path"
    path = "test/path"
    section = "Paths"

    config = configparser.ConfigParser()
    config[section] = {}
    with open(config_file, "w") as f:
        config.write(f)

    add_path_to_config(path_name, path, config_file, section)
    config.read(config_file)
    assert config[section][path_name] == path


def test_fetch_path_from_config():
    config_file = "test_config.ini"
    section = "Paths"
    path_name = "test_path"
    path = "test/path"

    config = configparser.ConfigParser()
    config[section] = {path_name: path}
    with open(config_file, "w") as f:
        config.write(f)

    fetched_path = fetch_path_from_config(section, path_name, config_file)
    assert fetched_path == path

if __name__ == "__main__":
    pytest.main()
