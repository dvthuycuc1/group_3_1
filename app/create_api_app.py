from fastapi import APIRouter, FastAPI

from app.api.import_data_sql import router as import_data_sql_router
from app.api.predict_energy_data import router as predict_energy_data_router

def __create_apt_router():
    api_router = APIRouter()
    api_router.include_router(import_data_sql_router)
    api_router.include_router(predict_energy_data_router)

    return api_router


def create_api_app():
    app = FastAPI(root_path='/')
    app.include_router(__create_apt_router())
    return app

