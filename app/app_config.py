from pydantic import BaseModel
from os import environ
from typing import Dict
import json

CONFIG_FILE = ".env"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILES_COL = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']

class AppConfig(BaseModel):
    DATABASE: str = 'qn_prediction'
    USER: str = 'postgres'
    PASSWORD: str = 'password123'
    HOST: str = "localhost"
    PORT: str = '5432'

    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 2177


def load_dict_config(config_value: str) -> Dict:
    return json.loads(config_value)


def load_config() -> AppConfig:
    config = AppConfig(**environ)
    return config


CONFIG = load_config()
