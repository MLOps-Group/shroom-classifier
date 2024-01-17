# syntax = docker/dockerfile:experimental
# Base Image
FROM --platform=linux/amd64 python:3.8-slim

EXPOSE $PORT
EXPOSE $WANDB_API_KEY
WORKDIR /app

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* 


COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY shroom_classifier/ shroom_classifier/
COPY Makefile Makefile
COPY configs/ configs/

# Set the working directory
WORKDIR /
# ARG WANDB_API_KEY
# ENV WANDB_API_KEY=$WANDB_API_KEY
# Install required system packages
# RUN --mount=type=cache,target=~/.cache/pip pip install -r requirements.txt
RUN make docker_requirements
ENV CONFIGSERVER_PORT=8080

# CMD exec uvicorn simple_fastapi_app:app --port $PORT --host 0.0.0.0 --workers 1
# CMD make run_app port=$PORT
# CMD python -u shroom_classifier/app/main.py
CMD gunicorn -w 4 -k uvicorn.workers.UvicornWorker shroom_classifier.app.main:app
