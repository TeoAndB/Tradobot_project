.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = tradobot
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Update requirements
update_requirements:
	$ pipreqs src/

## Make Dataset
# Yahoo Finance:
data: requirements

	$(PYTHON_INTERPRETER) src/data/make_dataset_YahooDownloader.py data/raw data/processed

data_1min: requirements

	$(PYTHON_INTERPRETER) src/data/make_dataset_1Min.py data/external data/processed

prep_data_1min: requirements

	$(PYTHON_INTERPRETER) src/data/preprocess_dataset_1Min.py data/interim data/processed

# ALPACA API
data_alpaca: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset_AlpacaAPI.py data/processed/data_Alpaca-API

	#$(PYTHON_INTERPRETER) src/data/make_dataset_YahooDownloader.py data/raw data/processed


# Train the model
train_DQN_return: requirements
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d2.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d2.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d2.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d3.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d3.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d3.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d4.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d4.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d4.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d5.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d5.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d5.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d6.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d6.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d6.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d7.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d7.py data/processed models
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_d7.py data/processed models



train_DQN_return_alternative: requirements
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_return_alternate_years.py data/processed models

validate_DQN_return: requirements
	$(PYTHON_INTERPRETER) src/models/validate_DQN_model_expReplay.py data/processed models


train_DQN_sharpe: requirements
	$(PYTHON_INTERPRETER) src/models/train_DQNmodel_sharpe.py data/processed models

# Train the model
run_baseline: requirements
	$(PYTHON_INTERPRETER) src/models/run_baseline_model.py data/processed models

# Visualize explainability files
visualize_explainability: requirements
	$(PYTHON_INTERPRETER) src/visualization/visualize_explainability.py reports/results_DQN reports/explainability_figures

# Visualize losses files
visualize_losses: requirements
	$(PYTHON_INTERPRETER) src/visualization/visualize_losses.py reports/tables/results_DQN reports/explainability_figures

visualize_profits: requirements
	$(PYTHON_INTERPRETER) src/visualization/visualize_profits.py reports/results_DQN reports/explainability_figures

# Visualize losses files
visualize_profits_model_comparison: requirements
	$(PYTHON_INTERPRETER) src/visualization/visualize_profits_model_comparison.py reports/tables/results_DQN reports/profits_model_comparison

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
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
	| LC_ALL='C' sort --ignore-case \
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
