#  docker build . -t limpbot/droid-slam:v1 --build-arg UID=$(id -u) --build-arg GID=$(id -g)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/home/appuser/miniconda3/bin:${PATH}"

# Install Droid SLAM dependencies
RUN conda install python=3.9

RUN conda env create -f environment.yml

RUN pip install 'pillow<7'

RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth