FROM --platform=linux/amd64 python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
# COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY shroom_classifier/ shroom_classifier/
COPY Makefile Makefile

# COPY shroom_classifier/app/main.py fastapi_app.py
# COPY shroom_classifier/app/Procfile_main Procfile

# RUN make deployment_requirements
RUN pip install --upgrade pip
# RUN --mount=type=cache,target=/root/.cache pip install -r ./requirements.txt
RUN pip install -r ./requirements.txt
RUN pip install -e .