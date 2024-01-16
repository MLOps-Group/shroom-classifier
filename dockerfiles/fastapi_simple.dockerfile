FROM --platform=linux/amd64 python:3.8-slim
# FROM pytorch/pytorch:latest

EXPOSE $PORT 
EXPOSE $WANDB_API_KEY

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
# RUN pip install python-multipart
RUN pip install wandb

# COPY requirements.txt requirements.txt
# COPY requirements_dev.txt requirements_dev.txt
# COPY pyproject.toml pyproject.toml
# COPY shroom_classifier/ shroom_classifier/
# COPY Makefile Makefile

COPY shroom_classifier/app/simple.py simple_fastapi_app.py

# RUN make docker_requirements

# RUN wandb login $WANDB_API_KEY
CMD exec uvicorn simple_fastapi_app:app --port $PORT --host 0.0.0.0 --workers 1
