import typer
import od3d.io
from omegaconf import OmegaConf
from pathlib import Path
import logging
from od3d.benchmark import bench_single_method_local, bench_single_method_local_separate_venv, bench_single_method_local_docker, bench_single_method_torque, bench_single_method_slurm
app = typer.Typer()

@app.command()
def multiple(benchmark: str = typer.Option('pascal3d_nemo', '-b', '--benchmark'),
        ablation: str = typer.Option(None, '-a', '--ablation'),
        platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.DEBUG)

    file_fpath = Path(__file__).parent.resolve()
    config_dir_rel = "../../../config"
    config_dir_abs = file_fpath.joinpath(config_dir_rel)
    ablations_root_dir = config_dir_abs.joinpath("ablations")

    if ablation is None:
        cfgs = [od3d.io.load_hierarchical_config(benchmark=benchmark, platform=platform)]
    else:
        cfgs = []
        # create one config per ablation
        ablation_dir = ablations_root_dir.joinpath(ablation)
        for ablation_file_fpath in ablation_dir.iterdir():
            ablation_fpath_rel = str(ablation_file_fpath.relative_to(ablations_root_dir).with_suffix(''))
            cfgs.append(od3d.io.load_hierarchical_config(benchmark=benchmark, platform=platform, ablation=ablation_fpath_rel))

    # create one config per method
    methods_cfgs = []
    for cfg in cfgs:
        methods_keys = cfg.method.keys()
        for key in methods_keys:
            method_cfg = cfg.copy()
            method_cfg.method = cfg.method[key]
            method_cfg_exists = False
            for prev_method_cfg in methods_cfgs:
                if method_cfg == prev_method_cfg:
                    method_cfg_exists = True
            if not method_cfg_exists:
                methods_cfgs.append(method_cfg)

    print(f"{len(methods_cfgs)} configs with single method.")
    for method_cfg in methods_cfgs:
        if method_cfg.platform.link == 'local':
            bench_single_method_local(method_cfg)
        elif method_cfg.platform.link == 'local-separate-venv':
            bench_single_method_local_separate_venv(method_cfg)
        elif method_cfg.platform.link == 'local-docker':
            bench_single_method_local_docker(method_cfg)
        elif method_cfg.platform.link == 'torque':
            bench_single_method_torque(method_cfg)
        elif method_cfg.platform.link == 'slurm':
            bench_single_method_slurm(method_cfg)

@app.command()
def single_local(config_fpath: str = typer.Option(None, '-c', '--config')):
    logging.basicConfig(level=logging.DEBUG)
    method_cfg = OmegaConf.load(config_fpath)
    bench_single_method_local(method_cfg)


@app.command()
def test(benchmark: str = typer.Option('timeseries_internal', '-b', '--benchmark'),
         mode: str = typer.Option('local', '-m', '--mode'),
         tasks: str = typer.Option(None, '-t', '--tasks'),
         constraint: str = typer.Option('4h8c', '-c', '--constraint'),
         frameworks: str = typer.Option(None, '-f', '--frameworks'),
         localcode: str = typer.Option(None, '-l', '--localcode')):
    logging.basicConfig(level=logging.DEBUG)
    logger.info("test")