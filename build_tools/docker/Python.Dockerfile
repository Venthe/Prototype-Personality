FROM docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean & \
    apt update && \
    apt install --assume-yes python3.11=3.11.0~rc1-1~22.04 python3-pip=22.0.2+dfsg-1ubuntu0.4

ENV PIP_CACHE_DIR=/cache/pip
RUN mkdir --parents /cache/pip
ADD requirements.txt .
RUN --mount=type=cache,target=/cache/pip \
    python3.11 -m pip install --cache-dir=/cache/pip -r requirements.txt --ignore-installed

ADD main.py .

ENTRYPOINT ["python3.11"]
CMD ["main.py"]

# ENTRYPOINT [ "bash" ]