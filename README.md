# NeMo

## Installation

### Install

`python3 -m venv venv_od3d`   
`source venv_od3d/bin/activate`   
`pip3 install pip --upgrade`   
CUDA_HOME=/misc/software/cuda/cuda-11.7
PATH=${CUDA_HOME}/bin:${PATH}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
export FORCE_CUDA=1
export CUDA_HOME
export LD_LIBRARY_PATH
export FORCE_CUDA
export TORCH_CUDA_ARCH_LIST
./misc/software/cuda/add_environment_cuda11.7.sh

`pip install -U fvcore`
`pip install -U iopath`
pip install Cython
export PATH
export LD_LIBRARY_PATH
export CUDA_HOME
pip install torch
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

1. From local

    `pip install -e .`

2. From github

    `pip install git+https://github.com/Generative-Vision-Robust-Learning/od3d.git`

### Clone

1. Repository   
`git clone git@github.com:Generative-Vision-Robust-Learning/od3d.git`  

2. Submodules   
`git submodule update --init --recursive`

## Usage

### Images


1. OD3D  

`docker build -f docker/Dockerfile -t limpbot/od3d:v1 --build-arg UID=$(id -u) --build-arg GID=$(id DD-g) .`

2. Droid SLAM

`docker build -f third_party/envs/DROID-SLAM/Dockerfile -t limpbot/droid-slam:v1 --build-arg UID=$(id -u) --build-arg GID=$(id -g) third_party/envs/DROID-SLAM`

### Dataset 

For each dataset there are three steps required to set it up

1), `od3d dataset setup -d co3d`

--> Downloading all files from the raw dataset.

2), `od3d dataset preprocess-meta -d co3d`

--> Reading all files from the raw dataset and saving meta files in the dataset preprocess directory.

3), `od3d dataset preprocess -d co3d`

--> Preprocessing like creating the clean point cloud etc.

To make the dataset avilable on another platform, e.g. slurm, use

`od3d dataset rsync -s local -t slurm -d co3d`

### Benchmark

To evaluate one method use

`od3d bench multiple -b co3d_nemo -p slurm`.

You can evaluate multiple methods, by specyfing an ablation directory, e.g. `nemo_old`

`od3d bench multiple -b co3d_nemo -p slurm -a nemo_old`.

To see the current status on slurm use

`od3d bench status-slurm`.

To stop a job running on slurm use

`od3d bench stop-slurm -j <job-name>`.

### Coordinate Frames

The semantic axes of a camera are
   - x: right (pytorch3d: left)
   - y: bottom (pytorch3d: top)
   - z: front (pytorch3d: front)

The semantic axes of an object are
   - x: left (pytorch3d: left) 
   - y: back (pytorch3d: top)
   - z: top (pytorch3d: front)

This leads to the `cam_tform4x4_obj` for a camera looking straight at the front of an object of:  
[  [1,  0,  0,     0],  
   [0,  0, -1,     0],   
   [0, 1,  0, +dist],  
   [0,  0,  0,     0],
]

Open3d uses as default `cam_tform4x4_obj`:   
[    [1,  0,  0,     0],  
   [0,  -1, 0,     0],   
   [0, 0,  -1, +dist],  
   [0,  0,  0,     0],
]


## Roadmap


### Models


- [x] NeMo (Shipped: *Dec 08 2022*)
- [x] NeMo-6D (Shipped: *Dec 09 2022*)
- [x] ResNet50-General (Shipped: *Dec 09 2022*)
- [ ] NeMo-Cls
- [ ] Domain adaptation (from synthetic to real)
- [ ] StarMap
- [ ] PASCAL3D-Specific
- [ ] Faster R-CNN
- [ ] Mask R-CNN
- [ ] Transformers
- [ ] VoGe Renderer

### Datasets

- [x] PASCAL3D+ (Shipped: *Dec 06 2022*)
- [x] Occluded PASCAL3D+ (Shipped: *Dec 06 2022*)
- [x] 6D training data (Shipped: *Dec 07 2022*)
- [ ] OOD-CV
- [ ] SyntheticPASCAL3D+
- [ ] ObjectNet3D

### Misc

- [x] Rewrite training and evaluate entry point (Shipped: *Dec 11 2022*)
- [x] Project page (Shipped: *Dec 11 2022*)
- [ ] Configuration hierarchy
- [ ] Visualization tools
- [ ] Inference demo
- [ ] Save predictions for reuse

## Citation

```
   @inproceedings{wang2021nemo,
      title={NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation},
      author={Angtian Wang and Adam Kortylewski and Alan Yuille},
      booktitle={International Conference on Learning Representations},
      year={2021},
      url={https://openreview.net/forum?id=pmj131uIL9H}
   }
```

```
   @software{nemo_code_2022,
      title={Neural Mesh Models for 3D Reasoning},
      author={Ma, Wufei and Jesslen, Artur and Wang, Angtian},
      month={12},
      year={2022},
      url={https://github.com/wufeim/NeMo},
      version={1.0.0}
   }
```

## Previous Work

---

* [NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation](https://openreview.net/forum?id=pmj131uIL9H) (ICLR 2021)
* [Robust Category-Level 6D Pose Estimation with Coarse-to-Fine Rendering of Neural Features](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_29) (ECCV 2022)

## Acknowledgements

In this project, we borrow codes from several other repos:

* :code:`NeMo` by Angtian Wang in `Angtian/NeMo <https://github.com/Angtian/NeMo>`_
* :code:`DMTet` by NVIDIA in `nv-tlabs/GET3D <https://github.com/nv-tlabs/GET3D>`_
* :code:`torch_utils` by NVIDIA in `nv-tlabs/GET3D <https://github.com/nv-tlabs/GET3D>`_
* :code:`uni_rep` by NVIDIA in `nv-tlabs/GET3D <https://github.com/nv-tlabs/GET3D>`_
* :code:`dnnlib` by NVIDIA in `nv-tlabs/GET3D <https://github.com/nv-tlabs/GET3D>`_
