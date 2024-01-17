# FROM --platform=linux/amd64 python:3.8-slim
FROM gcr.io/shroom-classifier-project/gcp_requirements_image

EXPOSE $PORT 
EXPOSE $WANDB_API_KEY

WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY shroom_classifier/ shroom_classifier/
COPY Makefile Makefile

COPY shroom_classifier/app/main.py fastapi_app.py

# # RUN make deployment_requirements
# RUN pip install --upgrade pip
# # RUN --mount=type=cache,target=/root/.cache pip install -r ./requirements.txt
# RUN pip install -r ./requirements.txt
# RUN pip install -e .

RUN echo "web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker fastapi_app:app" > Procfile
CMD exec uvicorn fastapi_app:app --port $PORT --host 0.0.0.0 --workers 1
