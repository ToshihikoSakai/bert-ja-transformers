FROM ubuntu:18.04
#FROM python-gpu
#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 
LABEL maintainer="toshiwork5630@gmail.com"
LABEL version="1.0"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git

RUN pip3 install --upgrade pip
#RUN pip3 install torch

#RUN pip3 install \
#    transformers \
#    sentencepiece \
#    pandas

# Install python modules.
#COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Set Japanese environment
RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen ja_JP.UTF-8 && \
    echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# Set alias for python3
RUN echo "alias python=python3" >> $HOME/.bashrc && \
    echo "alias pip=pip3" >> $HOME/.bashrc


WORKDIR /work


CMD ["/bin/bash"]