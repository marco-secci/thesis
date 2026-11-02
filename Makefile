# Single entry point of the project. Every target runs inside the container:
# nothing is installed on the host.
#
# The engine is podman, in rootless mode: no sudo needed, no need to be in the
# `docker` group, and files written in the bind-mount stay owned by the host
# user (root in the container = host user, under rootless).
# To use Docker instead of podman: `make ENGINE=docker ...` — this however
# requires the user to be in the `docker` group.

SHELL := /bin/bash

ENGINE  ?= podman
IMAGE   ?= thesis-py:latest
IMAGE_R ?= thesis-r:latest
IMAGE_TEX ?= thesis-tex:latest
REPO    := $(CURDIR)

RUN := $(ENGINE) run --rm -v $(REPO):/work -w /work \
         -e PYTHONPATH=/work/src -e MPLCONFIGDIR=/tmp/mpl -e HOME=/tmp \
         $(IMAGE)

.DEFAULT_GOAL := help

.PHONY: help build build-r build-tex shell manifest ingest clean-data panel eda \
        retention glm dash-overdisp baselines models forecast population figures \
        export-legacy thesis test all shiny

help:  ## List the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build:  ## Build the Python image
	$(ENGINE) build --target py -t $(IMAGE) -f Dockerfile .

build-r:  ## Build the R image (slow; only needed for the dashboard)
	$(ENGINE) build --target r -t $(IMAGE_R) -f Dockerfile .

build-tex:  ## Build the LaTeX image (slow; only needed for `make thesis`)
	$(ENGINE) build --target tex -t $(IMAGE_TEX) -f Dockerfile .

shell:  ## Interactive shell inside the container
	$(ENGINE) run --rm -it -v $(REPO):/work -w /work \
	  -e PYTHONPATH=/work/src -e MPLCONFIGDIR=/tmp/mpl -e HOME=/tmp $(IMAGE) bash

manifest:  ## Regenerate the sha256 manifest of the raw files
	$(RUN) python -m manifest

ingest:  ## 27 XLSX -> data/interim/events_raw.parquet + reconciliation
	$(RUN) python -m ingest

clean-data:  ## events_raw -> data/processed/donations.parquet
	$(RUN) python -m clean

panel:  ## donations -> data/processed/donor_year.parquet
	$(RUN) python -m panel

eda:  ## Exploratory figures and tables in reports/
	$(RUN) python -m eda

baselines:  ## Rolling-origin backtest of the baselines
	$(RUN) python -m baselines

models:  ## Full model comparison (baselines, aggregate series, per donor)
	$(RUN) python -m backtest_models --sims 30

population:  ## Donation rates per resident population (Eurostat denominator)
	$(RUN) python -m population

retention:  ## Ch. 3: retention and survival curves by cohort
	$(RUN) python -m retention

glm:  ## Ch. 4: inferential GLM at the donor-year grain, IRR table
	$(RUN) python -m glm_inference

dash-overdisp:  ## App. B: Poisson/NB AIC comparison at the dashboard grain
	$(RUN) python -m dash_overdisp

figures:  ## Regenerate all thesis figures in thesis/img/
	$(RUN) python -m figures_tesi

export-legacy:  ## Compatibility CSVs for the Shiny dashboard
	$(RUN) python -m export_legacy

forecast:  ## 2026-2031 forecast with the selected model
	$(RUN) python -m forecast

thesis:  ## Compile thesis/Tesi.tex -> thesis/Tesi.pdf
	$(ENGINE) run --rm -v $(REPO):/work -w /work/thesis -e HOME=/tmp $(IMAGE_TEX) \
	  latexmk -pdf -interaction=nonstopmode -halt-on-error Tesi.tex

test:  ## pytest on the invariants
	$(RUN) pytest

all: ingest clean-data panel eda retention glm dash-overdisp baselines models \
     forecast population figures test  ## Full pipeline

shiny: export-legacy  ## Start the dashboard at http://localhost:8181
	$(ENGINE) run --rm -it -p 8181:8181 -v $(REPO):/work -w /work -e HOME=/tmp $(IMAGE_R)
