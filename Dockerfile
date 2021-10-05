FROM ubuntu:18.04
#FROM nvidia/cuda

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git

RUN pip3 install --upgrade pip

# Install python modules
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Set Japanese environment
RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen ja_JP.UTF-8 && \
    echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# Set GPU:0 device
ENV CUDA_VISIBLE_DEVICES 0

# Set alias for python3
RUN echo "alias python=python3" >> $HOME/.bashrc && \
    echo "alias pip=pip3" >> $HOME/.bashrc

# pytorch install(CPU)
RUN pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install pytorch_pretrained_bert

WORKDIR /work


CMD ["/bin/bash"]