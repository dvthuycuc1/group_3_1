FROM python:3.10.13-bookworm

# create directory for the app user
RUN mkdir -p /home/app

# create the appropriate directories
ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/staticfiles
RUN mkdir $APP_HOME/mediafiles
WORKDIR $APP_HOME

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install psycopg2 dependencies
RUN apt-get update && apt-get install libcurl4 libcurl4-openssl-dev -y

# lint
RUN pip install --upgrade pip
RUN pip install gunicorn
RUN pip install flake8
RUN flake8 --ignore=E501,F401 .

# install dependencies
COPY ./requirements.txt $APP_HOME
RUN pip install -r requirements.txt
RUN pip install flask[async]
COPY ./data /usr/src/app

# copy project
COPY . $APP_HOME
