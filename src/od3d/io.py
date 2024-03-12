import logging
logger = logging.getLogger(__name__)
import urllib.request
import time
import sys
from pathlib import Path
import tarfile
import zipfile
import shutil
import os
import gdown
from hydra import compose, initialize, initialize_config_dir

from omegaconf import DictConfig, OmegaConf
import json
from typing import Dict
import subprocess

import importlib


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
def download(url: str, fpath: Path):
    if fpath.exists():
        logging.warning(f"File {fpath} already exists. Skip download {url}.")
    else:
        if not fpath.parent.exists():
            fpath.parent.mkdir(parents=True)

        if "google.com" in url:
            gdown.download(url=url, output=str(fpath), fuzzy=True)
        else:
            urllib.request.urlretrieve(url, fpath, reporthook)


def unzip(fpath: Path, dst: Path):
    with zipfile.ZipFile(fpath, 'r') as zip_ref:
        zip_ref.extractall(dst)
    os.remove(fpath)

def untar(fpath: Path, dst: Path):
    file = tarfile.open(fpath)
    file.extractall(dst)
    file.close()
    os.remove(fpath)

def move_dir(src: Path, dst: Path):
    for _fpath in src.iterdir():
        shutil.move(_fpath, dst)
    shutil.rmtree(src)

def rm_dir(path: Path):
    try:
        logger.info(f'removing directory {path}')
        path.unlink()
        shutil.rmtree(path)
    except Exception as e:
        logger.warning(e)

from tqdm import tqdm
def load_multiple_hierarchical_configs(benchmark="defaults", platform="local", multiple_ablations=[], multiple_overrides=[]):
    config_dir_rel = "../../config"
    cfgs= []
    with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
        for a, ablations in tqdm(enumerate(multiple_ablations)):
            # logger.info(ablations)
            if len(multiple_overrides) > a:
                overrides = multiple_overrides[a]
            else:
                overrides = []
            overrides = [f"+ablations/{Path(ablation).parent}={Path(ablation).stem}" for ablation in ablations] + [
                "platform=" + platform] + overrides
            cfg = compose(config_name=benchmark, overrides=overrides)
            #cfg.ablation_name = '_'.join(
            #    [cfg[key] for key in list(filter(lambda k: k.startswith('ablation_name_'), cfg.keys()))])

            from omegaconf import open_dict
            with open_dict(cfg):
                cfg.ablation_name = '_'.join([ablation.stem for ablation in ablations])
            logger.info(cfg.ablation_name)

            cfgs.append(cfg)
    return cfgs

def load_hierarchical_config(benchmark="defaults", platform="local", ablations=[], overrides=[]):
    config_dir_rel = "../../config"
    with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
        overrides = [f"+ablations/{Path(ablation).parent}={Path(ablation).stem}" for ablation in ablations] + ["platform=" + platform] + overrides
        cfg = compose(config_name=benchmark, overrides=overrides)

        #if ablations is None:
        #    cfg = compose(config_name=benchmark, overrides=["platform=" + platform] + overrides)
        #else:
        #    cfg = compose(config_name=benchmark, overrides=["+ablations=" + ablation, "platform=" + platform] + overrides)
    return cfg

def read_config_intern(rfpath: Path, benchmark="defaults", platform="local", overrides=[]):
    config_dir_rel = "../../config"

    try:
        with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
            cfg = compose(config_name=benchmark, overrides=[f"+{rfpath.parent}=" + str(rfpath.stem), "platform=" + platform] + overrides)

        cfg = cfg
        for key in str(rfpath.parent).split('/'):
            cfg = cfg.get(key)
    except hydra.errors.ConfigCompositionException as e:
        fpath = Path("config").joinpath(rfpath)
        logger.warning(e)
        logger.warning('trying to read with OmegaConf')
        cfg = OmegaConf.load(fpath)

    return cfg


import hydra.errors

def read_config_extern(fpath: Path):
    try:
        with initialize_config_dir(config_dir=str(fpath.parent.absolute()), job_name="test_app_extern"):
            cfg = compose(config_name=fpath.stem)
    except hydra.errors.ConfigCompositionException as e:
        logger.warning(e)
        logger.warning('trying to read with OmegaConf')
        cfg = OmegaConf.load(fpath)

    return cfg

def write_config_to_json_file(config: DictConfig, fpath: Path):
    write_json(config=dict(config), fpath=fpath)

def write_json(config: Dict, fpath: Path):
    fpath.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(fpath.expanduser(), "w") as outfile:
        json.dump(config, outfile)

def read_json(fpath: Path):
    with open(fpath.expanduser(), 'r') as openfile:
        config = json.load(openfile)
    return config

def run_cmd(cmd, logger, live=False, background=False):
    if logger is not None:
        logger.info(f'Run command {cmd}')
    if live:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)

        for line in process.stdout:
            # Print or process the live output as needed
            logger.info(line)
            # print(line, end='')

        # Wait for the subprocess to complete
        process.wait()

        # Retrieve the return code of the subprocess
        return_code = process.returncode
        logger.info(f'Return Code {return_code}')

    else:
        if not background:
            res = subprocess.run(cmd, capture_output=True, shell=True)
            if logger is not None:
                logger.info(res.stdout.decode("utf-8"))
                logger.info(res.stderr.decode("utf-8"))
            return res.stdout.decode("utf-8")
        else:
            child = RunCmdBackgroundProcess(cmd, os.getpid())
            child_proc = Process(target=child.run_child)
            child_proc.daemon = True
            child_proc.start()

            #subprocess.run(cmd,  capture_output=False, shell=True, stderr=None, stdout=None, start_new_session=False)

import psutil
from multiprocessing import Process
from time import sleep


class RunCmdBackgroundProcess(object):
    def __init__(self, command, parent_pid):
        """
        @type parent_pid: int
        @type command: str
        """
        self._child = None
        self._cmd = command
        self._parent = psutil.Process(pid=parent_pid)

    def run_child(self):
        """
        Start a child process by running self._cmd.
        Wait until the parent process (self._parent) has died, then kill the
        child.
        """
        self._child = subprocess.Popen([self._cmd], shell=True)
        try:
            while self._parent.status() == psutil.STATUS_RUNNING or self._parent.status() == psutil.STATUS_SLEEPING:
                sleep(1)
        except psutil.NoSuchProcess:
            pass
        finally:
            self._child.terminate()


def read_str_from_file(fpath: Path):
    with open(fpath, 'r') as file:
        data = file.read().rstrip()
    return data

def write_str_to_file(fpath: Path, text: str):
    with open(fpath, "w") as file:
        file.write(text)

from typing import List
import tempfile

def write_dict_as_yaml(fpath: Path, _dict: Dict):
    conf = OmegaConf.create(_dict)
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, 'w') as fp: #  tempfile.NamedTemporaryFile()
        OmegaConf.save(config=conf, f=fp.name)
def read_dict_from_yaml(fpath: Path):
    with open(fpath, 'r') as fp:
        loaded = OmegaConf.load(fp.name)
    return loaded

def write_list_as_yaml(fpath: Path, _list: List[str]):
    conf = OmegaConf.create(_list)
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, 'w') as fp: #  tempfile.NamedTemporaryFile()
        OmegaConf.save(config=conf, f=fp.name)

def read_list_from_yaml(fpath: Path):
    with open(fpath, 'r') as fp:
        loaded = OmegaConf.load(fp.name)
    return loaded


def get_obj_from_config(*args, config: DictConfig, **kwargs):
    class_name_split = config.class_name.split('.')
    module_name = '.'.join(class_name_split[:-1])
    class_name = class_name_split[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(*args, **{**kwargs, **config.kwargs})
