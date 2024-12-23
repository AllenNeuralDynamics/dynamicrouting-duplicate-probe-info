# hash:sha256:6ee112feb247e11fb27a5632549f39ed9be5bd81cef120fb7044af21c2f0c39e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-session==0.3.13 \
    polars==1.17.1 \
    tqdm==4.67.1 \
    universal-pathlib==0.1.4 \
    zarr==2.18.0
