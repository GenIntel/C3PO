FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

#FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


#PATH_OD3D =
#VENV_NAME=venv310
#PATH_VENV =
#
## Setup Repository
#if [[ -d "{cfg.platform.path_od3d}" ]]; then
#    echo "OD3D is already cloned to {cfg.platform.path_od3d}."
#else
#    git clone {cfg.platform.url_od3d} {cfg.platform.path_od3d}
#fi
#
#cd {cfg.platform.path_od3d}
#git pull {cfg.platform.url_od3d}
#
## Install OD3D in venv
#VENV_NAME=venv310
#export VENV_NAME
#if [[ -d "${{VENV_NAME}}" ]]; then
#    echo "Venv already exists at {cfg.platform.path_od3d}/${{VENV_NAME}}."
#    source {cfg.platform.path_od3d}/${{VENV_NAME}}/bin/activate
#else
#    echo "Creating venv at {cfg.platform.path_od3d}/${{VENV_NAME}}."
#    python3 -m venv {cfg.platform.path_od3d}/${{VENV_NAME}}
#    source {cfg.platform.path_od3d}/${{VENV_NAME}}/bin/activate
#fi
#RUN python3 -m venv venv_od3d
#RUN source venv
RUN pip install pip --upgrade
RUN pip install torch
RUN FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
#RUN pip3 install git+https://github.com/Generative-Vision-Robust-Learning/od3d.git

#RUN git clone https://github.com/Generative-Vision-Robust-Learning/od3d.git

#WORKDIR od3d

