# Base image
FROM  --platform=linux/amd64 python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY shroom-classifier/ shroom-classifier/
COPY data/ data/
COPY Makefile Makefile

WORKDIR /
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "shroom_classifier/train_model.py"]