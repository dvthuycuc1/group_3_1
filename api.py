"""
Author:
Date: 23-10-29
Desc: main script to run
"""
import uvicorn
from app.app_config import CONFIG
from app.create_api_app import create_api_app

api_app = create_api_app()

if __name__ == '__main__':
    # print(CONFIG.FASTAPI_HOST)
    uvicorn.run(api_app)
