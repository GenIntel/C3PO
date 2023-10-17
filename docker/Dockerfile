#docker build -f docker/Dockerfile -t limpbot/od3d:v1 --build-arg UID=$(id -u) --build-arg GID=$(id DD-g) .

# note: devel is required to have nvcc
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

#FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

#FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

#
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

# Get the basic Linux packages upgraded and get the development tools setup
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y wget \
    build-essential \
    manpages-dev \
    ca-certificates \
    g++ \
    git \
    libglfw3-dev \
    libgles2-mesa-dev \
    && apt-get clean  \
    && gcc --version

# rm -rf /var/lib/apt/lists/* \

RUN apt-get install -y python3.10 python3-pip
RUN pip install pip --upgrade
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV CUDA_HOME=/usr/local/cuda
RUN pip install torch

RUN pip install faiss-gpu
RUN pip install fvcore
RUN pip install iopath
# ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
#RUN apt-get install -y gcc-10* libstdc++-10*

#ENV NVIDIA_VISIBLE_DIVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
# # 3.5;
# torch.cuda.get_arch_list()
# CUDA_FLAGS =  --expt-relaxed-constexpr -Xcompiler=-fPIC   -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11
#ENV CMAKE_CXX_FLAGS:STRING=-std=c++17
RUN FORCE_CUDA=1 pip install git+https://github.com/facebookresearch/pytorch3d.git
# @stable # note dont use stable version because it throws c++17 missing error

# CUDNN=cudnn-linux-x86_64-8.9.5.29_cuda11-archive
#cp ${CUDNN}/include/cudnn*.h ${CUDA_HOME}/include
#cp ${CUDNN}/lib/libcudnn* ${CUDA_HOME}/lib64
#chmod a+r ${CUDA_HOME}/include/cudnn*.h ${CUDA_HOME}/lib64/libcudnn*

#CUDA_HOME=/scratch/sommerl/cudas/cuda-11.8
#export CUDA_HOME
#PATH=${PATH}:${CUDA_HOME}/bin
#LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64 # ${LD_LIBRARY_PATH}:
#export PATH
#export LD_LIBRARY_PATH

#  fvcore -c iopath -c conda-forge fvcore iopath
#RUN CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

ADD . /od3d/

WORKDIR od3d

RUN pip install -e .


RUN apt-get install -y docker.io

RUN docker build -f third_party/envs/DROID-SLAM/Dockerfile -t limpbot/droid-slam:v1 --build-arg UID=$(id -u) --build-arg GID=$(id -g) third_party/envs/DROID-SLAM

#RUN pip install git+https://github.com/Generative-Vision-Robust-Learning/od3d.git
## FORCE_CUDA=1

#RUN git clone https://github.com/Generative-Vision-Robust-Learning/od3d.git
#WORKDIR od3d
#RUN pip install -e
