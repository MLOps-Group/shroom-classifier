# Base Image
FROM --platform=linux/amd64 python:3.8-slim

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
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY
# Install required system packages
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN make docker_requirements

# Train the model
ENTRYPOINT ["python", "-u", "shroom_classifier/train_model.py"]