#  docker build -f third_party/envs/Zero-Shot-Pose/Dockerfile -t limpbot/zsp:v1 third_party/zero-shot-pose

#--build-arg ssh_prv_key="$(cat ~/.ssh/id_rsa)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)" --squash
#--build-arg UID=$(id -u) --build-arg GID=$(id -g)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install wget -y

WORKDIR /root

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

# Install Droid SLAM dependencies
RUN conda install python=3.9


#RUN apt-get install git -y

ADD . /root/zsp

WORKDIR /root/zsp/

RUN wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth

##ADD ~/.ssh/id_rsa /root/.ssh/id_rsa
##
#RUN mkdir -p /root/.ssh
#RUN chmod 0700 /root/.ssh
#RUN echo "$ssh_prv_key"
#
#RUN echo "SHA256:uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s" > /root/.ssh/known_hosts
#
### Add the keys and set permissions
#RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa
#RUN echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub
#RUN chmod 600 /root/.ssh/id_rsa
#RUN chmod 600 /root/.ssh/id_rsa.pub
#
##RUN  echo "    IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config
#RUN echo "Host remotehost\n\tStrictHostKeyChecking no\n"  >> /etc/ssh/ssh_config
##RUN echo "Host remotehost\n\tStrictHostKeyChecking no\n"  >> /home/user/.ssh/config
#
#RUN git clone git@github.com:limpbot/zero-shot-pose.git
#
## Remove SSH keys
#RUN rm -rf /root/.ssh/

RUN conda env create -f environment.yml

RUN conda activate zsp

RUN pip install 'pillow<7'

RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

RUN pip install Flask


ENTRYPOINT ["FLASK_ENV=development", "FLASK_APP=flask.py", "flask", "run"]