FROM docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean & \
    apt update && \
    apt install --assume-yes git=1:2.34.1-1ubuntu1.11 python3.11=3.11.0~rc1-1~22.04 python3-pip=22.0.2+dfsg-1ubuntu0.4

ENV PIP_CACHE_DIR=/cache/pip
RUN mkdir --parents /cache/pip
ADD requirements.txt .
# Ugly hack to workaround the AutoGPTQ being stupid with numpy and torch
# Ugly hack to workadound the Git failing when installing transformers
RUN --mount=type=cache,target=/cache/pip git config --global http.version HTTP/1.1 && \
    python3.11 -m pip install --cache-dir=/cache/pip \
    wheel \
    numpy \
    torch==2.4.1+cu124 \
    protobuf \
    sentencepiece \
    pillow \
    flask \
    configparser \
    git+https://github.com/huggingface/optimum@2c0476eda1398b9a81cb966c817a460ed6e53413 \
    accelerate>=0.26.0 \
    git+https://github.com/huggingface/transformers@v4.45.2 --extra-index-url https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/cache/pip python3.11 -m pip install --cache-dir=/cache/pip -r requirements.txt

ADD main.py .

EXPOSE 5000

ENTRYPOINT ["python3.11"]
CMD ["main.py"]
