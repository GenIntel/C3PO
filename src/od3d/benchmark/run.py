import logging
logger = logging.getLogger(__name__)
import subprocess
from omegaconf import DictConfig, OmegaConf
from od3d.benchmark.benchmark import OD3D_Benchmark
from od3d.io import run_cmd


def bench_single_method_local(config: DictConfig):
    benchmark = OD3D_Benchmark(config=config)
    benchmark.run()

def bench_single_method_local_separate_venv(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d in separate virtual environment
    # 3. from virtual environment: run od3d bench single -f `path-to-config`
    # TODO
    raise NotImplementedError

def bench_single_method_local_docker(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d in docker image
    # 3. from inside docker: run od3d bench single -f `path-to-config`
    # TODO
    raise NotImplementedError
def torque_run_method_or_cmd(cfg: DictConfig, cmd=None):
    # 1. save config
    # 2. setup od3d on torque
    # 3. execute script with command: run od3d bench single -f `path-to-config`

    if cmd is None:
        job_name = cfg.run_name
    else:
        job_name = cmd.replace(' ', '_')

    from pathlib import Path
    local_tmp_config_fpath = Path(cfg.platform_local.path_home).joinpath('tmp', f'config_{job_name}.yaml') # .resolve() # .resolve()
    if not local_tmp_config_fpath.resolve().parent.exists():
        local_tmp_config_fpath.parent.mkdir(parents=True)
    with open(local_tmp_config_fpath.resolve(), 'w') as fp:
        OmegaConf.save(config=cfg, f=fp)
    local_tmp_script_fpath = Path(cfg.platform_local.path_home).joinpath('tmp', f'run_{job_name}.sh') # .resolve()
    if not local_tmp_script_fpath.parent.exists():
        local_tmp_script_fpath.parent.mkdir(parents=True)


    remote_tmp_config_fpath = Path(cfg.platform.path_home).joinpath('tmp', f'config_{job_name}.yaml')
    remote_tmp_script_fpath_parent = Path(cfg.platform.path_home).joinpath('tmp')
    remote_tmp_script_fpath = remote_tmp_script_fpath_parent.joinpath(f'run_{job_name}.sh')

    if cmd is None:
        cmd = f'od3d bench single-local -c {remote_tmp_config_fpath}'

    with open(local_tmp_script_fpath, 'w') as rsh:

        gpu_count = cfg.platform.gpu_count
        gpu_mem_in_gb = cfg.platform.gpu_mem_in_gb
        node_count = 1
        cpu_count = cfg.platform.cpu_count
        ram = cfg.platform.ram
        walltime = cfg.platform.walltime

        if gpu_count > 0:
            if gpu_mem_in_gb > 16:
                if gpu_mem_in_gb > 24:
                    logger.warning(f'GPU memory of {gpu_mem_in_gb} GB is too large. Using 24 GB instead.')
                #gpu_mem_cfg_str = ':nvidiaMin24GB'
                gpu_mem_cfg_str = ':nvidiaRTX3090'
            elif gpu_mem_in_gb > 12:
                # gpu_mem_cfg_str = ':nvidiaMin16GB'
                gpu_mem_cfg_str = ':nvidiaP100'
            elif gpu_mem_in_gb > 11:
                # gpu_mem_cfg_str = ':nvidiaMin12GB'
                gpu_mem_cfg_str = ':nvidiaP100'
            elif gpu_mem_in_gb > 10:
                # gpu_mem_cfg_str = ':nvidiaMin11GB'
                gpu_mem_cfg_str = ':nvidiaRTX2080Ti'
            elif gpu_mem_in_gb > 6:
                # gpu_mem_cfg_str = ':nvidiaMin8GB'
                gpu_mem_cfg_str = ':nvidiaRTX2080Ti'
            elif gpu_mem_in_gb > 0:
                gpu_mem_cfg_str = ':nvidiaRTX2080Ti'
            else:
                gpu_mem_cfg_str = ':nvidiaRTX2080Ti'
        else:
            gpu_mem_cfg_str = ""

        gpu_cfg_str = f':gpus={gpu_count}' if gpu_count > 0 else ""
        cuda_cfg_str = f':nvidiaMinCC75' if gpu_count > 0 else ""

        if cfg.platform.pull_od3d:
            pull_od3d_cmds_str = f'''
git fetch 
git checkout {cfg.branch}
git pull
            '''
        else:
            pull_od3d_cmds_str = ''

        if cfg.platform.pull_od3d_submodules:
            pull_od3d_submodules_cmds_str = f'''
git submodule init
git submodule update
git submodule foreach 'git fetch origin; git checkout $(git rev-parse --abbrev-ref HEAD); git reset --hard origin/$(git rev-parse --abbrev-ref HEAD); git submodule update --recursive; git clean -dfx'
            '''
        else:
            pull_od3d_submodules_cmds_str = ''

        if cfg.platform.install_od3d:
            # headless open3d rendering infeasible due to requirements
            # https://github.com/isl-org/Open3D/blob/main/util/install_deps_ubuntu.sh (most likely clang version)
            install_od3d_cmds_str = f'''
pip install pip --upgrade
pip install wheel
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -e {cfg.platform.path_od3d}
# pip uninstall open3d -y
# rm -rf Open3D
# git clone --recursive --branch v0.17.0 https://github.com/isl-org/Open3D.git
# cd Open3D/
# mkdir build && cd build
# cmake -DBUILD_FILAMENT_FROM_SOURCE=ON \
#       -DBUILD_SHARED_LIBS=ON \
#       -DENABLE_HEADLESS_RENDERING=ON \
#       -DBUILD_GUI=OFF \
#       -DBUILD_WEBRTC=OFF \
#       -DUSE_SYSTEM_GLEW=OFF \
#       -DUSE_SYSTEM_GLFW=OFF ..
# make -j$(nproc)
# make install
# make install-pip-package
# cd ../..
            '''
        else:
            install_od3d_cmds_str = ''

        script_as_string = f'''#!/bin/bash
#PBS -N {job_name}
#PBS -S /bin/bash
#PBS -l nodes={node_count}:ppn={cpu_count}{gpu_cfg_str}{gpu_mem_cfg_str}{cuda_cfg_str},mem={ram},walltime={walltime}
#PBS -q {cfg.platform.queue}
#PBS -m a
#PBS -M {cfg.platform.username}@informatik.uni-freiburg.de
#PBS -j oe

# For interactive jobs: #PBS -I
# For array jobs: #PBS -t START-END[%SIMULTANEOUS]

echo $(curl google.com)

CUDA_HOME={cfg.platform.path_cuda}
PATH=${{CUDA_HOME}}/bin:${{PATH}}
LD_LIBRARY_PATH=${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH}}
export PATH
export LD_LIBRARY_PATH
export CUDA_HOME

echo PATH=${{PATH}}
echo LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}
echo CUDA_HOME=${{CUDA_HOME}}

# Setup Repository
if [[ -d "{cfg.platform.path_od3d}" ]]; then
    echo "OD3D is already cloned to {cfg.platform.path_od3d}."
else
    git clone {cfg.platform.url_od3d} {cfg.platform.path_od3d}
fi

while [[ -e "{cfg.platform.path_od3d}/installing.txt" ]]; do
    sleep 3  
    echo "waiting for installing.txt file to disappear."
done

touch "{cfg.platform.path_od3d}/installing.txt"

cd {cfg.platform.path_od3d}

{pull_od3d_cmds_str}
{pull_od3d_submodules_cmds_str}

# Install OD3D in venv
VENV_NAME=venv_od3d
export VENV_NAME
if [[ -d "${{VENV_NAME}}" ]]; then
    echo "Venv already exists at {cfg.platform.path_od3d}/${{VENV_NAME}}."
    source {cfg.platform.path_od3d}/${{VENV_NAME}}/bin/activate
else
    echo "Creating venv at {cfg.platform.path_od3d}/${{VENV_NAME}}."
    python3 -m venv {cfg.platform.path_od3d}/${{VENV_NAME}}
    source {cfg.platform.path_od3d}/${{VENV_NAME}}/bin/activate
fi

{install_od3d_cmds_str}

rm "{cfg.platform.path_od3d}/installing.txt"

od3d debug hello-world

{cmd}

#PYTHONUNBUFFERED=1 
#CUDA_VISIBLE_DEVICES=1

exit 0
        '''
        rsh.write(script_as_string)
    #subprocess.run(f'scp {tmp_script_fpath} torque:{tmp_script_fpath}', capture_output=True, shell=True)
    #subprocess.run(f'scp {tmp_config_fpath} torque:{tmp_config_fpath}', capture_output=True, shell=True)

    if not cfg.platform.shared_home_with_local:
        run_cmd(f'ssh torque "mkdir -p {remote_tmp_script_fpath_parent}"', logger=None)
        run_cmd(f'scp {local_tmp_script_fpath} torque:{remote_tmp_script_fpath}', logger=None)
        run_cmd(f'scp {local_tmp_config_fpath} torque:{remote_tmp_config_fpath}', logger=None)
    run_cmd(f'ssh torque "cd torque_jobs && qsub {remote_tmp_script_fpath}"', logger=None)

def slurm_run_method_or_cmd(cfg: DictConfig, cmd=None):
    # 1. save config
    # 2. setup od3d on slurm
    # 3. execute script with command: run od3d bench single -f `path-to-config`

    if cmd is None:
        job_name = cfg.run_name
    else:
        job_name = cmd.replace(' ', '_')

    from pathlib import Path
    local_tmp_config_fpath = Path(cfg.platform_local.path_home).joinpath('tmp', f'config_{job_name}.yaml') # .resolve() # .resolve()
    if not local_tmp_config_fpath.resolve().parent.exists():
        local_tmp_config_fpath.parent.mkdir(parents=True)
    with open(local_tmp_config_fpath.resolve(), 'w') as fp:
        OmegaConf.save(config=cfg, f=fp)
    local_tmp_script_fpath = Path(cfg.platform_local.path_home).joinpath('tmp', f'run_{job_name}.sh') # .resolve()
    if not local_tmp_script_fpath.parent.exists():
        local_tmp_script_fpath.parent.mkdir(parents=True)

    remote_tmp_config_fpath = Path(cfg.platform.path_home).joinpath('tmp', f'config_{job_name}.yaml')
    remote_tmp_script_fpath_parent = Path(cfg.platform.path_home).joinpath('tmp')
    remote_tmp_script_fpath = remote_tmp_script_fpath_parent.joinpath(f'run_{job_name}.sh')

    if cmd is None:
        cmd = f'od3d bench single-local -c {remote_tmp_config_fpath}'

    with open(local_tmp_script_fpath, 'w') as rsh:
        gpu_count = cfg.platform.gpu_count
        gpu_mem_in_gb = cfg.platform.gpu_mem_in_gb
        node_count = 1
        cpu_count = cfg.platform.cpu_count
        ram = cfg.platform.ram
        walltime = cfg.platform.walltime

        if cfg.platform.pull_od3d:
            pull_od3d_cmds_str = f'''
git fetch 
git checkout {cfg.branch}
git pull
            '''
        else:
            pull_od3d_cmds_str = ''

        if cfg.platform.pull_od3d_submodules:
            pull_od3d_submodules_cmds_str = f'''
git submodule init
git submodule update
git submodule foreach 'git fetch origin; git checkout $(git rev-parse --abbrev-ref HEAD); git reset --hard origin/$(git rev-parse --abbrev-ref HEAD); git submodule update --recursive; git clean -dfx'
            '''
        else:
            pull_od3d_submodules_cmds_str = ''

        if cfg.platform.install_od3d:
            # headless open3d rendering infeasible due to requirements
            # https://github.com/isl-org/Open3D/blob/main/util/install_deps_ubuntu.sh (most likely clang version)
            install_od3d_cmds_str = f'''
pip install pip --upgrade
pip install wheel
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -e {cfg.platform.path_od3d}
# pip uninstall open3d -y
# rm -rf Open3D
# git clone --recursive --branch v0.17.0 https://github.com/isl-org/Open3D.git
# cd Open3D/
# mkdir build && cd build
# cmake -DBUILD_FILAMENT_FROM_SOURCE=ON \
#       -DBUILD_SHARED_LIBS=ON \
#       -DENABLE_HEADLESS_RENDERING=ON \
#       -DBUILD_GUI=OFF \
#       -DBUILD_WEBRTC=OFF \
#       -DUSE_SYSTEM_GLEW=OFF \
#       -DUSE_SYSTEM_GLFW=OFF ..
# make -j$(nproc)
# make install
# make install-pip-package
# cd ../..
            '''
        else:
            install_od3d_cmds_str = ''

        partition = cfg.get("platform").get("partition", None)
        partition_cfg_str = f'#SBATCH --partition {partition}' if partition is not None else ''
        script_as_string = f'''#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --nodes {node_count}
#SBATCH --ntasks-per-node 1
#SBATCH --time {walltime}
#SBATCH --cpus-per-task {cpu_count}
#SBATCH --gres gpu:{gpu_count}
#SBATCH --mem {ram}
#SBATCH -o {cfg.platform.path_home}/slurm_jobs/%x_%j.o # x=job_name j=job_id
#SBATCH --mail-type=FAIL  # END,FAIL # (recive mails about end and timeouts/crashes of your job)
{partition_cfg_str}

CUDA_HOME={cfg.platform.path_cuda}
PATH=${{CUDA_HOME}}/bin:${{PATH}}
LD_LIBRARY_PATH=${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH}}
export PATH
export LD_LIBRARY_PATH
export CUDA_HOME

HTTP_PROXY=http://tfsquid.informatik.intra.uni-freiburg.de:8080
HTTPS_PROXY=http://tfsquid.informatik.intra.uni-freiburg.de:8080
export HTTP_PROXY
export HTTPS_PROXY

echo PATH=${{PATH}}
echo LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}
echo CUDA_HOME=${{CUDA_HOME}}

# Setup Repository
if [[ -d "{cfg.platform.path_od3d}" ]]; then
    echo "OD3D is already cloned to {cfg.platform.path_od3d}."
else
    git clone {cfg.platform.url_od3d} {cfg.platform.path_od3d}
fi

while [[ -e "{cfg.platform.path_od3d}/installing.txt" ]]; do
    sleep 3  
    echo "waiting for installing.txt file to disappear."
done

touch "{cfg.platform.path_od3d}/installing.txt"

cd {cfg.platform.path_od3d}

{pull_od3d_cmds_str}
{pull_od3d_submodules_cmds_str}

# Install OD3D in venv
VENV_NAME=venv_od3d
export VENV_NAME
if [[ -d "${{VENV_NAME}}" ]]; then
    echo "Venv already exists at {cfg.platform.path_od3d}/${{VENV_NAME}}."
    source {cfg.platform.path_od3d}/${{VENV_NAME}}/bin/activate
else
    echo "Creating venv at {cfg.platform.path_od3d}/${{VENV_NAME}}."
    python3 -m venv {cfg.platform.path_od3d}/${{VENV_NAME}}
    source {cfg.platform.path_od3d}/${{VENV_NAME}}/bin/activate
fi

{install_od3d_cmds_str}

rm "{cfg.platform.path_od3d}/installing.txt"

od3d debug hello-world

{cmd}

exit 0
        '''
        rsh.write(script_as_string)
    if not cfg.platform.shared_home_with_local:
        run_cmd(f'ssh slurm "mkdir -p {remote_tmp_script_fpath_parent}"', logger=None)
        run_cmd(f'scp {local_tmp_script_fpath} slurm:{remote_tmp_script_fpath}', logger=None)
        run_cmd(f'scp {local_tmp_config_fpath} slurm:{remote_tmp_config_fpath}', logger=None)
    run_cmd(f'ssh slurm "sbatch {remote_tmp_script_fpath}"', logger=None)

    # ws_allocate {cfg.platform.ws_name} 100 -m sommerl@informatik.uni-freiburg.de
    # ws_allocate od3d 100 -m sommerl@informatik.uni-freiburg.de # /work/dlclarge1/sommerl-od3d
    # ws_list
    # TODO
    # raise NotImplementedError