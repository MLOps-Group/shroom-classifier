# Base Image
FROM --platform=linux/amd64 python:3.8-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# copy the project files to the working directory
# COPY data.dvc code/data.dvc
COPY requirements.txt code/requirements.txt
COPY requirements_dev.txt code/requirements_dev.txt
COPY pyproject.toml code/pyproject.toml
COPY shroom_classifier/ code/shroom_classifier/
COPY Makefile code/Makefile
COPY configs/ code/configs/

# Set the working directory
WORKDIR /code

# Install required system packages
RUN make docker_requirements

# Get data
# RUN dvc init --no-scm
# RUN dvc remote add -d myremote gs://shroom_bucket

# Train the model
ENTRYPOINT ["python", "-u", "shroom_classifier/train_model.py"]