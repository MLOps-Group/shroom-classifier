## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the same name exists
## See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
.PHONY: *

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = shroom_classifier
PYTHON_VERSION := 3.8
PYTHON_INTERPRETER = python

# GOOGLE CLOUD
PROJECT_ID = shroom-classifier-project
SECRET_NAME = wandb_api_key
SECRET_VERSION = latest
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install Developer Python Dependencies
dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install .["dev"]

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete




## Get coverage report
coverage:
	python -m pip install coverage
	coverage run -m pytest tests/
	coverage report -m

#################################################################################
# DEPLOYMENT RULES                                                              #
#################################################################################

## Run app
run_app: APP = simple
run_app: PORT = 8000
run_app: DOCKER = False
run_app:
	if [ $(DOCKER) = True ]; then \
		wandb docker-run -p $(PORT):$(PORT) -e PORT=$(PORT)  gcr.io/$(PROJECT_ID)/gcp_$(APP)_app; \
	else \
		uvicorn --reload --port $(PORT) shroom_classifier.app.$(APP):app; \
	fi
	# uvicorn --reload --port $(PORT) shroom_classifier.app.$(APP):app

## Get app from Google Cloud Container Registry
get_app: APP = simple
get_app:
	docker pull gcr.io/$(PROJECT_ID)/gcp_$(APP)_app

## Build docker image and push to Google Cloud Container Registry

build_app: APP = simple
build_app:
	docker build -t gcp_$(APP)_app . -f dockerfiles/fastapi_$(APP).dockerfile
	docker tag gcp_$(APP)_app gcr.io/$(PROJECT_ID)/gcp_$(APP)_app
	docker push gcr.io/$(PROJECT_ID)/gcp_$(APP)_app

## Deploy app to Google Cloud Run
deploy_app: APP = simple
deploy_app: SERVICE_NAME = shroom-classifier-app-v2
deploy_app: REGION = europe-west1
deploy_app: IMAGE = gcr.io/$(PROJECT_ID)/shroom_classifier-app
deploy_app: PORT = 8000
deploy_app: MEMORY = 2Gi
deploy_app: SECRET_NAME = wandb_api_key
deploy_app:
	gcloud run deploy $(APP)-app \
		--image gcr.io/$(PROJECT_ID)/gcp_$(APP)_app:latest \
		--platform managed \
		--region $(REGION) \
		--port $(PORT) \
		--memory $(MEMORY) \
		--allow-unauthenticated \
		--project $(PROJECT_ID) \
		--set-secrets WANDB_API_KEY=wandb_api_key:latest \
		--service-account app-handler@shroom-classifier-project.iam.gserviceaccount.com \
		--set-env-vars CLOUD_RUN=True \

create_requirements_image:
	docker build -t gcp_requirements_image . -f dockerfiles/requirements.dockerfile
	docker tag gcp_requirements_image gcr.io/$(PROJECT_ID)/gcp_requirements_image
	docker push gcr.io/$(PROJECT_ID)/gcp_requirements_image
	
		
## Requirements for deployment
deployment_requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install gunicorn --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install -e . --no-cache-dir

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process raw data into processed data
data:
	python $(PROJECT_NAME)/data/make_dataset.py


## Train model
train: config_file = train_default
train:
	python $(PROJECT_NAME)/train_model.py train_config=$(config_file)


#################################################################################
## Docker RULES                                                                 #
#################################################################################

## Docker requirements
docker_requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --no-cache-dir
	$(PYTHON_INTERPRETER) -m pip install -e . --no-cache-dir

## Build docker image
build_docker:
	docker build -t $(PROJECT_NAME)-train . -f dockerfiles/train_model.dockerfile

## Run docker image
run_docker: image = train
run_docker:
	docker run $(PROJECT_NAME)-$(image)



# Fetch API key from Google Cloud Secret Manager
get_api_key: SECRET_VERSION = latest
get_api_key: SECRET_NAME = wandb_api_key
get_api_key: PROJECT_ID = shroom-project-410914
get_api_key:
	export WANDB_API_KEY=$(gcloud secrets versions access ${SECRET_VERSION} --secret=${SECRET_NAME} --project=${PROJECT_ID} | base64 -d)
	echo "WANDB_API_KEY=$(WANDB_API_KEY)"

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')