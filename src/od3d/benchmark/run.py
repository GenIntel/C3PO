import subprocess
from omegaconf import DictConfig, OmegaConf
from od3d.benchmark.benchmark import OD3D_Benchmark

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
def bench_single_method_torque(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d on torque
    # 3. execute script with command: run od3d bench single -f `path-to-config`
    # TODO

    job_name = cfg.run_name

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
    remote_tmp_script_fpath = Path(cfg.platform.path_home).joinpath('tmp', f'run_{job_name}.sh')

    with open(local_tmp_script_fpath, 'w') as rsh:

        gpu_count = cfg.platform.gpu_count
        node_count = 1
        cpu_count = cfg.platform.cpu_count
        ram = cfg.platform.ram
        walltime = cfg.platform.walltime

        gpu_cfg_str = f':gpus={gpu_count}' if gpu_count > 0 else ""
        cuda_cfg_str = f':nvidiaMinCC75' if gpu_count > 0 else ""

        if cfg.platform.install_od3d:
            install_od3d_cmds_str = f'''
pip install pip --upgrade
pip install torch
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -e {cfg.platform.path_od3d}
            '''
        else:
            install_od3d_cmds_str = ''

        script_as_string = f'''#!/bin/bash
#PBS -N {job_name}
#PBS -S /bin/bash
#PBS -l nodes={node_count}:ppn={cpu_count}{gpu_cfg_str}{cuda_cfg_str},mem={ram},walltime={walltime}
#PBS -q default-cpu
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

cd {cfg.platform.path_od3d}
git pull {cfg.platform.url_od3d}

# Install OD3D in venv
VENV_NAME=venv310
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

od3d debug hello-world

od3d bench single-local -c {remote_tmp_config_fpath}
#PYTHONUNBUFFERED=1 
#CUDA_VISIBLE_DEVICES=1

exit 0
        '''
        rsh.write(script_as_string)
    #subprocess.run(f'scp {tmp_script_fpath} torque:{tmp_script_fpath}', capture_output=True, shell=True)
    #subprocess.run(f'scp {tmp_config_fpath} torque:{tmp_config_fpath}', capture_output=True, shell=True)
    subprocess.run(f'ssh torque "cd torque_jobs && qsub {remote_tmp_script_fpath}"', capture_output=True, shell=True)

def bench_single_method_slurm(cfg: DictConfig):
    # 1. save config
    # 2. setup od3d on slurm
    # 3. execute script with command: run od3d bench single -f `path-to-config`

    job_name = cfg.run_name
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
    remote_tmp_script_fpath = Path(cfg.platform.path_home).joinpath('tmp', f'run_{job_name}.sh')

    with open(local_tmp_script_fpath, 'w') as rsh:
        gpu_count = cfg.platform.gpu_count
        node_count = 1
        cpu_count = cfg.platform.cpu_count
        ram = cfg.platform.ram
        walltime = cfg.platform.walltime

        if cfg.platform.install_od3d:
            install_od3d_cmds_str = f'''
pip install pip --upgrade
pip install torch
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -e {cfg.platform.path_od3d}
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

cd {cfg.platform.path_od3d}
git pull {cfg.platform.url_od3d}

# Install OD3D in venv
VENV_NAME=venv310
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

od3d debug hello-world

od3d bench single-local -c {remote_tmp_config_fpath}

exit 0
        '''
        rsh.write(script_as_string)
    #subprocess.run(f'scp {remote_tmp_script_fpath} slurm:{remote_tmp_script_fpath}', capture_output=True, shell=True)
    #subprocess.run(f'scp {remote_tmp_config_fpath} slurm:{remote_tmp_config_fpath}', capture_output=True, shell=True)

    subprocess.run(f'ssh slurm "sbatch {remote_tmp_script_fpath}"', capture_output=True, shell=True)

    # ws_allocate {cfg.platform.ws_name} 100 -m sommerl@informatik.uni-freiburg.de
    # ws_allocate od3d 100 -m sommerl@informatik.uni-freiburg.de # /work/dlclarge1/sommerl-od3d
    # ws_list
    # TODO
    # raise NotImplementedError