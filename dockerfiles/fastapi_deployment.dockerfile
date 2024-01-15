FROM python:3.8-slim

EXPOSE $PORT



RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY


COPY shroom_classifier shroom_classifier
COPY Makefile Makefile

RUN make docker_requirements

# CMD exec uvicorn simple_fastapi_app:app --port $PORT --host 0.0.0.0 --workers 1
CMD make run_app
