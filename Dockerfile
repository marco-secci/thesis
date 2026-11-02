# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# Stage `py` — Python working environment. It is the one used by every
# Makefile target. The code is NOT copied into the image: the Makefile
# bind-mounts it, so an edit on the host is immediately visible inside.
# ─────────────────────────────────────────────────────────────────────────────
# Image names are qualified with the registry (`docker.io/library/...`):
# podman, unlike docker, has no default registry and a short name does not
# resolve.
FROM docker.io/library/python:3.12-slim AS py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /usr/local/bin/

WORKDIR /work

# Only the dependency files, so the layer is invalidated only when they change.
COPY pyproject.toml uv.lock README.md ./
# `--no-install-project` installs only the dependencies: the `thesis` package
# comes from the bind-mount via PYTHONPATH. Installing it here would leave in
# site-packages a copy frozen at build time, which would shadow the mounted
# one whenever PYTHONPATH is not set.
RUN uv sync --frozen --extra dev --no-install-project

ENV PYTHONPATH=/work/src
CMD ["bash"]

# ─────────────────────────────────────────────────────────────────────────────
# Stage `r` — only for running the Shiny dashboard in shiny_app/.
# It is not needed by any part of the data pipeline: it is built separately
# with `make build-r` because it is much heavier and slower than the py stage.
# ─────────────────────────────────────────────────────────────────────────────
FROM docker.io/library/r-base:4.4.1 AS r

# libuv1-dev is needed by `fs`, which shiny, bslib, DT and shinyWidgets
# depend on transitively: without it, those do not install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev \
        libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev \
        libtiff5-dev libjpeg-dev libuv1-dev pandoc \
    && rm -rf /var/lib/apt/lists/*

# `install.packages` does not return an error code when a package fails to
# install: the build would pass and the problem would only be found when
# starting the app. The explicit check below makes the build fail right away.
RUN R -q -e "\
      pkgs <- c('shiny','shinyWidgets','dplyr','tidyr','lubridate','ggplot2', \
                'scales','DT','MASS','readr'); \
      install.packages(pkgs, repos='https://cloud.r-project.org', Ncpus=4); \
      missing <- pkgs[!pkgs %in% rownames(installed.packages())]; \
      if (length(missing)) { \
        cat('PACCHETTI NON INSTALLATI:', paste(missing, collapse=', '), '\\n'); \
        quit(status=1) \
      }"

WORKDIR /work
EXPOSE 8181
CMD ["R", "-q", "-e", "shiny::runApp('shiny_app', host='0.0.0.0', port=8181)"]

# ── stage tex ────────────────────────────────────────────────────────────────
# Compiles the thesis. It is a separate stage because TeX Live weighs more than
# everything else put together: it is built only when needed
# (`make build-tex`), and is not part of `make build` or `make all`.
FROM docker.io/library/debian:bookworm-slim AS tex

RUN apt-get update && apt-get install -y --no-install-recommends \
      texlive-latex-base texlive-latex-recommended texlive-latex-extra \
      texlive-fonts-recommended texlive-fonts-extra texlive-lang-italian \
      texlive-science texlive-plain-generic lmodern latexmk \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work/thesis
