"""
Author:
Date: 23-10-29
Desc: main script to run
"""
import uvicorn
from app.app_config import CONFIG
from app.create_api_app import create_api_app
from flask_ui import app

api_app = create_api_app()

if __name__ == '__main__':
    # print(CONFIG.FASTAPI_HOST)
    # uvicorn.run(api_app, host=CONFIG.FASTAPI_HOST, port=CONFIG.FASTAPI_PORT)
    app.run(debug=True, port=5555)
