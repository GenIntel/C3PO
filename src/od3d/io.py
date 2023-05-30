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
from hydra import compose, initialize

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
            cfg = compose(config_name=benchmark, overrides=["ablations=" + ablation, "platform=" + platform] + overrides)
    return cfg


