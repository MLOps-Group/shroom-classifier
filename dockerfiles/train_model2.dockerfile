# Base Image
FROM --platform=linux/amd64 python:3.8-slim
# FROM google/cloud-sdk:alpine as gcloud
# WORKDIR /app
# ARG KEY_FILE_CONTENT
# RUN echo $KEY_FILE_CONTENT | gcloud auth activate-service-account --key-file=- \
#   && gsutil cp gs://shroom_bucket/ .

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Copy the service account key file
# COPY shroom-project-410914-7503fcf85328.json /tmp/shroom-project-410914-7503fcf85328.json

# # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/keyfile.json

# copy the project files to the working directory
# COPY data.dvc code/data.dvc
COPY data/ code/data/
COPY requirements.txt code/requirements.txt
COPY requirements_dev.txt code/requirements_dev.txt
COPY pyproject.toml code/pyproject.toml
COPY shroom_classifier/ code/shroom_classifier/
COPY Makefile code/Makefile
COPY configs/ code/configs/

# Set the working directory
WORKDIR /code

# Install required system packages
# RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN make docker_requirements


# Get data
# RUN dvc init --no-scm
# RUN dvc remote add -d remote_storage gs://shroom_bucket
# RUN dvc remote modify remote_storage version_aware true
# RUN dvc pull --force

# Train the model
ENTRYPOINT ["python", "-u", "shroom_classifier/train_model.py"]