MAKEFLAGS += --silent
DOCKER ?= $(shell which docker)
PIPENV ?= $(shell which pipenv)
PIP ?= $(shell which pip)
LOCAL_IMAGE ?= bco
IMAGE_URI ?= travisbyrum/bco-learner
TAG ?= latest

all: clean clean-pyc install

.PHONY: all

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

.PHONY: clean-pyc

clean:
	rm -rf build dist *.egg-info

.PHONY: clean

freeze:
	$(PIPENV) run pip freeze > requirements.txt

.PHONY: clean

docker-push: docker-build
	$(DOCKER) tag $(LOCAL_IMAGE):$(TAG) $(IMAGE_URI):$(TAG)
	$(DOCKER) push $(IMAGE_URI):$(TAG)
	docker push $(IMAGE_URI):$(TAG)

.PHONY: docker-push

docker-build:
	$(DOCKER) build --tag $(LOCAL_IMAGE):$(TAG) .

.PHONY: docker-build

run:
	$(DOCKER) run \
		-v `pwd`/training.py:/opt/bco/training.py \
		-v `pwd`/data:/opt/bco/data \
		$(LOCAL_IMAGE):$(TAG) train \
		--input malaria_phenotypes

.PHONY: run
