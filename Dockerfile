# Base image
FROM  --platform=linux/amd64 python:3.10-slim

# Install required system packages and Google Cloud SDK
RUN apt-get update
RUN apt-get install -y google-cloud-sdk
RUN apt-get install --no-install-recommends -y build-essential gcc
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app
# Authenticate with Google Cloud using the service account key
COPY shroom-project-410914-7503fcf85328.json /app/key.json
RUN gcloud auth activate-service-account --key-file=/app/key.json

# Install Google Cloud SDK
RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-366.0.0-linux-x86_64.tar.gz -o /tmp/google-cloud-sdk.tar.gz && \
    tar -xf /tmp/google-cloud-sdk.tar.gz -C /app && \
    /app/google-cloud-sdk/install.sh --quiet && \
    rm /tmp/google-cloud-sdk.tar.gz


COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY shroom_classifier/ shroom_classifier/
COPY Makefile Makefile


# Download data from Google Cloud Storage bucket
RUN gsutil -m cp -r gs://shroom_bucket/* /app/data/

WORKDIR /
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "/app/shroom_classifier/train_model.py"]