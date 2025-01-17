import pickle
import os

import yaml

# project root


# Using INI configuration file
# from configparser import ConfigParser

# config = ConfigParser()
# config.read(CONFIG_PATH)
# DB_PATH = str(config.get("PATHS", "DB_PATH"))
# MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))
# RANDOM_STATE = int(config.get("ML", "RANDOM_STATE"))

# # Doing the same with a YAML configuration file
# import yaml
#
# Get the current script directory
current_dir = os.path.dirname(__file__)

# Construct the path to the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

ROOT_DIR = parent_dir
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')

# Construct the path to config.yml in the parent directory
config_path = os.path.join(parent_dir, "config.yml")
# MODEL_PATH = os.path.join(parent_dir, "models", "model.pkl")


with open(config_path, "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    DB_PATH = str(config_yaml['paths']['db_path'])
    MODEL_PATH = str(config_yaml['paths']["model_path"])
    RANDOM_STATE = int(config_yaml["ml"]["random_state"])

    TRACKING_URI = config_yaml['mlflow']['tracking_uri']
    EXPERIMENT_NAME = config_yaml['mlflow']['experiment_name']
    MODEL_NAME = config_yaml['mlflow']['model_name']
    MODEL_VERSION = config_yaml['mlflow']['model_version']

# SQLite requires the absolute path
# DB_PATH = os.path.abspath(DB_PATH)
DB_PATH = os.path.join(ROOT_DIR, os.path.normpath(DB_PATH))


def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model
