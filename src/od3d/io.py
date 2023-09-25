import logging
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


def load_hierarchical_config(benchmark="defaults", platform="local", ablation=None, overrides=[]):
    config_dir_rel = "../../config"
    with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
        if ablation is None:
            cfg = compose(config_name=benchmark, overrides=["platform=" + platform] + overrides)
        else:
            cfg = compose(config_name=benchmark, overrides=["+ablations=" + ablation, "platform=" + platform] + overrides)
    return cfg

def read_config_extern(fpath: Path):
    with initialize_config_dir(config_dir=str(fpath.parent), job_name="test_app_extern"):
        cfg = compose(config_name=fpath.stem)
    return cfg

def write_config_to_json_file(config: DictConfig, fpath: Path):
    write_json(config=dict(config), fpath=fpath)

def write_json(config: Dict, fpath: Path):
    fpath.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(fpath.expanduser(), "w") as outfile:
        json.dump(config, outfile)

def load_json(fpath: Path):
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
            subprocess.run(cmd, capture_output=False, shell=True)


def read_str_from_file(fpath: Path):
    with open(fpath, 'r') as file:
        data = file.read().rstrip()
    return data

def write_str_to_file(fpath: Path, text: str):
    with open(fpath, "w") as file:
        file.write(text)

def get_obj_from_config(*args, config: DictConfig, **kwargs):
    class_name_split = config.class_name.split('.')
    module_name = '.'.join(class_name_split[:-1])
    class_name = class_name_split[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(*args, **{**kwargs, **config.kwargs})
