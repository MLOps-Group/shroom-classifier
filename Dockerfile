#Worked before
#RUN apt-get update && \
#    apt-get install -y curl && \
#    curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-366.0.0-linux-x86_64.tar.gz -o /tmp/google-cloud-sdk.tar.gz && \
#    tar -xzf /tmp/google-cloud-sdk.tar.gz -C /tmp && \
#    /tmp/google-cloud-sdk/install.sh --quiet && \
#    rm -rf /tmp/google-cloud-sdk && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*

# Base image
FROM  --platform=linux/amd64 python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY Makefile Makefile
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY shroom_classifier/ shroom_classifier/
COPY data.dvc data.dvc

WORKDIR /
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

RUN pip install dvc "dvc[gs]"
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://shroom_bucket/

RUN dvc init --no-scm
RUN dvc remote add -d remote_storage gs://shroom_bucket/
RUN dvc remote modify remote_storage version_aware true
RUN dvc pull --force

ENTRYPOINT ["python", "-u", "/app/shroom_classifier/train_model.py"]