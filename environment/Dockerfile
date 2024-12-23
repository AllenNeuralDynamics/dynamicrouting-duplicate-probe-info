# hash:sha256:fb1137e0d733f67c06ead3effba83325763059b1bb983f033d9f537229bc7bd7
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
    aind-session \
    polars==1.17.1 \
    tqdm \
    universal-pathlib==0.1.4 \
    zarr==2.18.0
