# NeMo

## Installation

1. From local

    `pip install -e .`

2. From github

    `pip install git+https://github.com/Generative-Vision-Robust-Learning/od3d.git`

## Usage

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
