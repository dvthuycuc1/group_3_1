from datetime import datetime
from flask import Flask

app = Flask(__name__) #__name__ == __main__
app.config['SECRET_KEY'] = 'abcd12345672dcth'
# app.config['UPLOAD_FOLDER']
# app.config['MAX_CONTENT_PATH']

from flask_ui import routes

