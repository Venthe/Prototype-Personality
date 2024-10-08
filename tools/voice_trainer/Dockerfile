FROM docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean & \
    apt update && \
    apt install --assume-yes software-properties-common=0.99.22.9 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install --assume-yes git=1:2.34.1-1ubuntu1.11 ffmpeg=7:4.4.2-0ubuntu0.22.04.1 python3.9=3.9.20-1+jammy1 python3-pip=22.0.2+dfsg-1ubuntu0.4 python3.9-distutils=3.9.20-1+jammy1

ENV PIP_CACHE_DIR=/cache/pip
RUN mkdir --parents /cache/pip
ADD requirements.txt .
RUN --mount=type=cache,target=/cache/pip \
    python3.9 -m pip install --cache-dir=/cache/pip -r requirements.txt \
    && python3.9 -m unidic download

ADD main.py .

ENTRYPOINT ["python3.9"]
CMD ["main.py"]