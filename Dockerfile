# syntax=docker/dockerfile:1
FROM python:3.7.13-bullseye
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
ENV FLASK_APP=detect.py
CMD [ "flask", "run", "--host=0.0.0.0"]