import logging
import multiprocessing
import os

import numpy as np
import scipy.io as sio
import wget
import pycocotools.mask
from tqdm import tqdm

from od3d.models.mesh_memory_map import MeshConverter
from od3d.utils import direction_calculator
from od3d.utils import prepare_pascal3d_sample
from od3d.utils.pascal3d_utils import CATEGORIES
from od3d.utils.pascal3d_utils import MESH_LEN

# from od3d.datasets.create_cuboid_mesh import create_meshes
from omegaconf import DictConfig
from pathlib import Path
import od3d.io

mesh_para_names = [
    "azimuth",
    "elevation",
    "theta",
    "distance",
    "focal",
    "principal",
    "viewport",
    "height",
    "width",
    "cad_index",
    "bbox",
]


def mask_to_rle(mask):
    mask = np.asfortranarray(mask)
    rle = {'counts': [], 'size': list(mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def rle_to_mask(rle):
    if isinstance(rle, np.ndarray):
        rle = rle[()]
    compressed_rle = pycocotools.mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    return pycocotools.mask.decode(compressed_rle).astype(np.uint8)


def download_pascal3d(config: DictConfig):
    path_pascal3d_raw = Path(config.path_pascal3d_raw)

    if path_pascal3d_raw.exists():
        logging.info(f"Found Pascal3D+ dataset at {path_pascal3d_raw}")
    else:
        logging.info(f"Downloading Pascal3D+ dataset at {path_pascal3d_raw}")
        fpath = path_pascal3d_raw.joinpath("pascal3d.zip")
        od3d.io.download(url=config.url_pascal3d_raw, fpath=fpath)
        od3d.io.unzip(fpath=fpath, dst=fpath.parent)
        od3d.io.move_dir(src=fpath.parent.joinpath(Path(config.url_pascal3d_raw).with_suffix("").name), dst=fpath.parent)

    """
    path_pascal3d_seg = Path(config.path_pascal3d_seg)
    if path_pascal3d_seg.exists():
        logging.info(f"Found Pascal3D+ segmentation dataset at {path_pascal3d_seg}")
    else:
        logging.info(f"Downloading Pascal3D+ segmentation dataset at {path_pascal3d_seg}")
        fpath = path_pascal3d_seg.joinpath("pascal3d_seg.zip")
        od3d.io.download(url=config.url_pascal3d_seg, fpath=fpath)
        od3d.io.unzip(fpath, fpath.parent)

    
    if hasattr(config, 'segmentation_masks') and len(getattr(config, 'segmentation_masks')) > 0:
        seg_data_path = config.seg_data_path
        if os.path.isdir(seg_data_path):
            print(f"Found segmentation data at {seg_data_path}")
        else:
            print(f"Generating segmentation data at {seg_data_path}")
            os.system(f'gdown {config.seg_data_url}')
            gdown.download(config.seg_data_url, output="Occluded_Vehicles.zip", fuzzy=True)
            os.system('unzip Occluded_Vehicles.zip')
            os.system('rm Occluded_Vehicles.zip')

            # Training
            for cate in CATEGORIES:
                img_path = os.path.join('Occluded_Vehicles', 'training', 'images', f'{cate}_raw')
                anno_path = os.path.join('Occluded_Vehicles', 'training', 'annotations', f'{cate}_raw')

                save_path = os.path.join(seg_data_path, 'train', cate)
                os.makedirs(save_path, exist_ok=True)

                filenames = [x for x in os.listdir(anno_path) if x.endswith('.npz')]
                for fname in tqdm(filenames, desc=f'training_{cate}'):
                    sz = Image.open(os.path.join(img_path, fname.split('.')[0]+'.JPEG')).size
                    annotation = np.load(os.path.join(anno_path, fname), allow_pickle=True)
                    try:
                        mask = pycocotools.mask.decode(pycocotools.mask.merge(pycocotools.mask.frPyObjects(annotation['mask'].tolist(), sz[1], sz[0])))
                        np.save(os.path.join(save_path, fname[:-4]), mask_to_rle(mask))
                    except Exception as e:
                        print(e)
                        continue

            # Validation
            for cate in CATEGORIES:
                for occ_level in [0, 1, 2, 3]:
                    img_path = os.path.join('Occluded_Vehicles', 'testing', 'images', f'{cate}FGL{occ_level}_BGL{occ_level}')
                    anno_path = os.path.join('Occluded_Vehicles', 'testing', 'annotations', f'{cate}FGL{occ_level}_BGL{occ_level}')

                    if occ_level == 0:
                        save_path = os.path.join(seg_data_path, 'val', f'{cate}')
                    else:
                        save_path = os.path.join(seg_data_path, 'val', f'{cate}FGL{occ_level}_BGL{occ_level}')
                    os.makedirs(save_path, exist_ok=True)

                    filenames = [x for x in os.listdir(anno_path) if x.endswith('.npz')]
                    for fname in tqdm(filenames, desc=f'val_{cate}FGL{occ_level}_BGL{occ_level}'):
                        sz = Image.open(os.path.join(img_path, fname.split('.')[0]+'.JPEG')).size
                        annotation = np.load(os.path.join(anno_path, fname))
                        try:
                            mask = pycocotools.mask.decode(pycocotools.mask.merge(pycocotools.mask.frPyObjects(annotation['mask'].tolist(), sz[1], sz[0])))
                            np.save(os.path.join(save_path, fname[:-4]), mask_to_rle(mask))
                        except Exception as e:
                            print(e)
                            continue
    else:
        print("Skipping segmentation data")
    """

def get_target_distances():
    ranges = np.linspace(4.0, 32.0, num=15)
    dists = np.zeros((14,), dtype=np.float32)
    for i in range(14):
        dists[i] = np.random.uniform(ranges[i], ranges[i + 1])
    return dists


def prepare_pascal3d(cfg):
    #workers = cfg.workers
    pascal3d_data_path = cfg.root_path
    dtd_raw_path = cfg.dtd_raw_path
    if os.path.isdir(pascal3d_data_path):
        print(f"Found prepared PASCAL3D+ dataset at {pascal3d_data_path}")
        return

    if cfg.pad_texture:
        dtd_mat = sio.loadmat(os.path.join(dtd_raw_path, "imdb", "imdb.mat"))
        images = dtd_mat["images"]
        dtd_ids = images[0, 0][0][0]
        dtd_filenames = images[0, 0][1][0]
        dtd_filenames = np.array([f[0] for f in dtd_filenames])
        dtd_splits = images[0, 0][2][0]
        dtd_classes = images[0, 0][3][0]

        train_dtd_filenames = []
        val_dtd_filenames = []
        for j, f, s in zip(dtd_ids, dtd_filenames, dtd_splits):
            if s == 1:
                train_dtd_filenames.append(f)
            elif s == 2:
                val_dtd_filenames.append(f)
        dtd_filenames = {"train": train_dtd_filenames, "val": val_dtd_filenames}

    if cfg.training_only:
        all_set_types = ["train"]
    elif cfg.evaluation_only:
        all_set_types = ["val"]
    else:
        all_set_types = ["train", "val"]

    tasks = []
    for set_type in all_set_types:
        save_root = os.path.join(pascal3d_data_path, set_type)
        os.makedirs(save_root, exist_ok=True)
        for occ in getattr(cfg.occ_levels, set_type):
            for cate in CATEGORIES:
                if cfg.pad_texture:
                    tasks.append([cfg, set_type, occ, cate, dtd_filenames[set_type]])
                else:
                    tasks.append([cfg, set_type, occ, cate, None])

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))

    total_samples, total_errors = {k: {} for k in all_set_types}, {
        k: {} for k in all_set_types
    }
    for (_err, _total, _set, _cate) in results:
        if _cate not in total_samples[_set]:
            total_samples[_set][_cate] = _total
            total_errors[_set][_cate] = _err
        else:
            total_samples[_set][_cate] += _total
            total_errors[_set][_cate] += _err
    for set_type in all_set_types:
        for cate in CATEGORIES:
            print(
                f"Prepared {total_samples[set_type][cate]} {set_type} samples for {cate}, "
                f"error rate {total_errors[set_type][cate]/total_samples[set_type][cate]*100:.2f}%"
            )


def worker(params):
    cfg, set_type, occ, cate, dtd_filenames = params
    pascal3d_raw_path = cfg.pascal3d_raw_path
    pascal3d_occ_raw_path = cfg.pascal3d_occ_raw_path
    dtd_raw_path = cfg.dtd_raw_path
    pascal3d_data_path = cfg.root_path
    save_root = os.path.join(pascal3d_data_path, set_type)
    prepare_seg = hasattr(cfg, 'segmentation_masks') and len(getattr(cfg, 'segmentation_masks')) > 0
    if prepare_seg:
        seg_data_path = cfg.seg_data_path

    this_size = cfg.image_sizes[cate]
    out_shape = [
        ((this_size[0] - 1) // 32 + 1) * 32,
        ((this_size[1] - 1) // 32 + 1) * 32,
    ]
    out_shape = [int(out_shape[0]), int(out_shape[1])]

    if occ == 0:
        data_name = ""
    else:
        data_name = f"FGL{occ}_BGL{occ}"
    save_image_path = os.path.join(save_root, "images", f"{cate}{data_name}")
    save_annotation_path = os.path.join(save_root, "annotations", f"{cate}{data_name}")
    save_list_path = os.path.join(save_root, "lists", f"{cate}{data_name}")
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_annotation_path, exist_ok=True)
    os.makedirs(save_list_path, exist_ok=True)

    list_dir = os.path.join(pascal3d_raw_path, "Image_sets")
    anno_dir = os.path.join(pascal3d_raw_path, "Annotations", f"{cate}_imagenet")
    if occ == 0:
        img_dir = os.path.join(pascal3d_raw_path, "Images", f"{cate}_imagenet")
        occ_mask_dir = ""
    else:
        img_dir = os.path.join(pascal3d_occ_raw_path, "images", f"{cate}{data_name}")
        occ_mask_dir = os.path.join(
            pascal3d_occ_raw_path, "annotations", f"{cate}{data_name}"
        )

    list_fname = os.path.join(list_dir, f"{cate}_imagenet_{set_type}.txt")
    with open(list_fname) as fp:
        image_names = fp.readlines()
    image_names = [x.strip() for x in image_names if x != "\n"]

    if cfg.mesh_path is not None:
        manager = MeshConverter(path=os.path.join(cfg.mesh_path, cate))
        direction_dicts = [direction_calculator(*t) for t in manager.loader]
    else:
        manager = direction_dicts = None

    num_errors = 0
    mesh_name_list = [[] for _ in range(MESH_LEN[cate])]
    for img_name in image_names:
        img_path = os.path.join(img_dir, f"{img_name}.JPEG")
        anno_path = os.path.join(anno_dir, f"{img_name}.mat")
        if prepare_seg:
            seg_mask_path = os.path.join(seg_data_path, set_type, cate, f'{img_name}.npy')
        else:
            seg_mask_path=None


        prepared_sample_names = prepare_pascal3d_sample(
            cate,
            img_name,
            img_path,
            anno_path,
            occ,
            save_image_path=save_image_path,
            save_annotation_path=save_annotation_path,
            out_shape=out_shape,
            prepare_mode=cfg.prepare_mode,
            augment_by_dist=(set_type == "train" and cfg.augment_by_dist),
            texture_filenames=dtd_filenames,
            texture_path=dtd_raw_path,
            single_mesh=cfg.single_mesh,
            mesh_manager=manager,
            direction_dicts=direction_dicts,
            seg_mask_path=seg_mask_path,
            center_and_resize=cfg.center_and_resize,
            skip_3d_anno=cfg.skip_3d_anno
        )

        if prepared_sample_names is None:
            num_errors += 1
            continue

        for (cad_index, sample_name) in prepared_sample_names:
            mesh_name_list[cad_index - 1].append(sample_name)

    for i, x in enumerate(mesh_name_list):
        with open(
            os.path.join(save_list_path, "mesh%02d" % (i + 1) + ".txt"), "w"
        ) as fl:
            fl.write("\n".join(x))

    return num_errors, len(image_names), set_type, cate