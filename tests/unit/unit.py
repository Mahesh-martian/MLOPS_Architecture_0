import configparser
import pandas as pd
from sklearn.model_selection import train_test_split
import pytest
from sklearn.preprocessing import StandardScaler

def ingest_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, test_size=0.2):
    train, test = train_test_split(data, test_size=test_size)
    return train, test

def add_paths_to_config(config_path, new_paths):
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'Paths' not in config.sections():
        config.add_section('Paths')
    for key, value in new_paths.items():
        config.set('Paths', key, value)
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def select_best_features(data, n_features):
    # Placeholder for feature selection logic
    selected_features = data.columns[:n_features]
    return data[selected_features]

def preprocess_data(train_data, test_data):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler

def test_preprocess_data():
    # Setup
    train_data = pd.DataFrame({'feature': [1, 2, 3]})
    test_data = pd.DataFrame({'feature': [4, 5]})

    # Execute
    train_scaled, test_scaled, scaler = preprocess_data(train_data, test_data)

    # Verify
    assert train_scaled.shape == train_data.shape
    assert test_scaled.shape == test_data.shape
    assert abs(train_scaled.mean()) < 1e-6  # Mean should be approximately 


def test_select_best_features():
    # Setup
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    })
    n_features = 2

    # Execute
    selected_data = select_best_features(data, n_features)

    # Verify
    assert selected_data.shape[1] == n_features
    assert list(selected_data.columns) == ['feature1', 'feature2']

def test_add_paths_to_config(tmp_path):
    # Setup
    config_path = tmp_path / "config.ini"
    config_path.write_text("[Paths]\nexisting_path=old_value\n")
    new_paths = {'new_path': 'new_value'}

    # Execute
    add_paths_to_config(config_path, new_paths)

    # Verify
    config = configparser.ConfigParser()
    config.read(config_path)
    assert config.get('Paths', 'existing_path') == 'old_value'
    assert config.get('Paths', 'new_path') == 'new_value'

def test_data_ingestion_and_split(tmp_path):
    # Setup
    data = pd.DataFrame({
        'feature': [1, 2, 3, 4, 5],
        'label': [0, 1, 0, 1, 0]
    })
    file_path = tmp_path / "data.csv"
    data.to_csv(file_path, index=False)

    # Execute
    ingested_data = ingest_data(file_path)
    train, test = split_data(ingested_data, test_size=0.4)

    # Verify
    assert not ingested_data.empty
    assert len(train) == 3
    assert len(test) == 2