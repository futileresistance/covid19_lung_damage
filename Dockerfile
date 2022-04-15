# syntax=docker/dockerfile:1
FROM python:3.7.13-bullseye
WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


COPY requirements.txt .
RUN  pip install -r requirements.txt
COPY . .
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
#ENV FLASK_APP=detect.py
COPY detect.py .
CMD ["python3", "detect.py"]
#RUN /opt/venv/bin/flask run --host 192.168.88.48 --port 8000
#CMD [ "$VIRTUAL_ENV/bin/flask", "run", "--host 0.0.0.0 --port 8000"]
