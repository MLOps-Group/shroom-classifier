# Base Image
FROM --platform=linux/amd64 python:3.8-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* \

# Set environment variables
# copy the project files to the working directory

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

# Train the model
ENTRYPOINT ["python", "-u", "shroom_classifier/train_model.py"]