## Object Detection 3D (OD3D)

### Install

#### Environment
```
python3 -m venv venv_od3d
source venv_od3d/bin/activate
pip3 install pip --upgrade
CUDA_HOME=/misc/software/cuda/cuda-11.7
export CUDA_HOME

pip install wheel # pytorch3d requires this
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### Install in Editable Mode
```
git submodule update --init --recursive
pip install -e .
```

#### Install in Non-editable mode

```
pip install git@github.com:Generative-Vision-Robust-Learning/od3d.git  
```

### Configure 

- Platform
  - `config/platform/local.yaml`  
  - `config/platform/torque.yaml`  
  - `config/platform/slurm.yaml`  
  - `~/.ssh/config` with `config/platform/ssh-config-template`  
    - Verify
      - `od3d platform run -p [torque|slurm]`
- Wandb
  - `wandb init`
- Credentials (optional)
    - `config/credentials/default.yaml`

    
### Dataset Setup

  1) Download
     - `od3d dataset setup -d [co3d|pascal3d|objectnet3d]`
  2) Extract Meta Data per Frame/Sequence (e.g. camera, quality, etc.)
     - `od3d dataset extract-meta -d [co3d|pascal3d|objectnet3d]`
  3) Preprocess (e.g. point cloud, mesh, masks, etc.)
     - `od3d dataset preprocess -d [co3d|pascal3d|objectnet3d]`

  - Visualize
    - `od3d dataset visualize -d [co3d|pascal3d|objectnet3d]`

  - Synchronize the target with the source platform
    - `od3d dataset rsync -s local -t slurm -d co3d`  
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s local -t torque -d co3d"`  
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s torque -t local -d objectnet3d"`
    - `od3d platform run -p slurm -c "od3d dataset rsync-preprocess -s torque -t local -d pascal3d"`  


### Benchmark

To evaluate one method use

  - `od3d bench multiple -b co3d_nemo -p slurm`.

You can evaluate multiple methods, by specyfing an ablation directory, e.g. `nemo_old`

  - `od3d bench multiple -b co3d_nemo -p slurm -a nemo_old`.

To see the current status on slurm use

  - `od3d bench status-slurm`.

To stop a job running on slurm use

  - `od3d bench stop-slurm -j <job-name>`.


### Documentation
- [Coordinate Frames](docs/coordinate_frames/README.md)