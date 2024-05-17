import logging

logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
from pathlib import Path
from od3d.cv.geometry.objects3d.meshes import Meshes

from od3d.datasets.pascal3d.enum import (
    PASCAL3D_CATEGORIES,
    MAP_CATEGORIES_OD3D_TO_PASCAL3D,
)
from od3d.datasets.pascal3d.frame import Pascal3DFrame
from od3d.datasets.frame import (
    OD3D_FRAME_MODALITIES,
    OD3D_FRAME_DEPTH_TYPES,
    OD3D_FRAME_KPTS2D_ANNOT_TYPES,
)
from od3d.datasets.pascal3d.enum import (
    PASCAL3D_SCALE_NORMALIZE_TO_REAL,
    PASCAL3D_CATEGORIES,
)
from typing import Dict, List
import shutil
import od3d.io

from od3d.datasets.ood_cv.frame import OOD_CV_FrameMeta, OOD_CV_Frame

from tqdm import tqdm

from od3d.data.ext_enum import ExtEnum
from od3d.datasets.object import OD3D_MESH_TYPES


class OOD_CV_SUBSETS(str, ExtEnum):
    CONTEXT = "context"
    POSE = "pose"
    SHAPE = "shape"
    TEXTURE = "texture"
    WEATHER = "weather"


class OOD_CV_CATEGORIES(str, ExtEnum):
    AEROPLANE = "aeroplane"
    BICYCLE = "bicycle"
    BOAT = "boat"
    # BOTTLE = "bottle"
    BUS = "bus"
    CAR = "car"
    CHAIR = "chair"
    DININGTABLE = "diningtable"
    MOTORBIKE = "motorbike"
    SOFA = "sofa"
    TRAIN = "train"
    # TVMONITOR = "tvmonitor"


class OOD_CV(OD3D_Dataset):
    all_categories = list(OOD_CV_CATEGORIES)
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_PASCAL3D
    frame_type = OOD_CV_Frame

    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        path_pascal3d_raw: Path,
        categories: List[PASCAL3D_CATEGORIES] = None,
        dict_nested_frames: Dict[str, Dict[str, List[str]]] = None,
        dict_nested_frames_ban: Dict[str, Dict[str, List[str]]] = None,
        transform=None,
        subset_fraction=1.0,
        index_shift=0,
    ):
        if categories is not None:
            if self.map_od3d_categories is not None:
                self.categories = [
                    self.map_od3d_categories.get(category, category)
                    if category not in self.all_categories
                    else category
                    for category in categories
                ]
                print(f"categories: {self.categories}")
            else:
                self.categories = categories
        else:
            self.categories = self.all_categories

        super().__init__(
            categories=categories,
            name=name,
            modalities=modalities,
            path_raw=path_raw,
            path_preprocess=path_preprocess,
            transform=transform,
            subset_fraction=subset_fraction,
            index_shift=index_shift,
            dict_nested_frames=dict_nested_frames,
            dict_nested_frames_ban=dict_nested_frames_ban,
        )

        self.path_pascal3d_raw = Path(path_pascal3d_raw)

    ##### DATASET PROPERTIES
    def get_frame_by_name_unique(self, name_unique):
        from od3d.datasets.object import (
            OD3D_CAM_TFORM_OBJ_TYPES,
            OD3D_FRAME_MASK_TYPES,
            OD3D_MESH_TYPES,
            OD3D_MESH_FEATS_TYPES,
            OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
            OD3D_TFROM_OBJ_TYPES,
        )

        #'name_unique', 'all_categories', 'depth_type', 'mask_type', 'cam_tform4x4_obj_type', 'kpts2d_annot_type', 'tform_obj_type', 'mesh_type', 'mesh_feats_type', and 'mesh_feats_dist_reduce_type' '''
        return self.frame_type(
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            modalities=self.modalities,
            name_unique=name_unique,
            all_categories=self.categories,
            path_meshes=self.path_pascal3d_raw,
            depth_type=OD3D_FRAME_DEPTH_TYPES.MESH,
            mask_type=OD3D_FRAME_MASK_TYPES.MESH,
            cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.META,
            kpts2d_annot_type=OD3D_FRAME_KPTS2D_ANNOT_TYPES.META,
            tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            mesh_type=OD3D_MESH_TYPES.META,
            mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
            mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.AVG,
        )

    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return OOD_CV(
            name=self.name,
            modalities=self.modalities,
            path_raw=self.path_raw,
            path_preprocess=self.path_preprocess,
            categories=self.categories,
            dict_nested_frames=dict_nested_frames,
            transform=self.transform,
            index_shift=self.index_shift,
            path_pascal3d_raw=self.path_pascal3d_raw,
        )

    # def get_item(self, item):
    #     frame_meta = OOD_CV_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta,
    #                                                                   name_unique=self.list_frames_unique[item])
    #     return OOD_CV_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
    #                         meta=frame_meta, modalities=self.modalities,
    #                         categories=self.categories)
    @staticmethod
    def setup(config: DictConfig):
        path_raw = Path(config.path_raw)

        # 3D Pose internal
        url = "https://drive.google.com/file/d/1djm2ugmk98__9jgL8Sqb_QUIIpC7e1ir/view"

        # 3D Pose Official
        # url = 'https://drive.google.com/file/d/1NlAPwPkriLgCcyhljBwb3xXxCyvhpPCj/view?usp=drive_link'

        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous OOD-CV")
            shutil.rmtree(path_raw)

        if path_raw.exists():
            logger.info(f"Found OOD-CV dataset at {path_raw}")
        else:
            logger.info(f"Downloading OOD-CV dataset at {path_raw}")
            path_raw.mkdir(parents=True, exist_ok=True)
            fpath = path_raw.joinpath(f"pose.zip")

            od3d.io.download(url=url, fpath=fpath)
            od3d.io.unzip(fpath, dst=path_raw)
            od3d.io.move_dir(src=path_raw.joinpath("ood_cv"), dst=path_raw)

    #### PREPROCESS META
    @staticmethod
    def extract_meta(config: DictConfig):
        MAP_OD3D_CATEGORIES = MAP_CATEGORIES_OD3D_TO_PASCAL3D
        subsets = config.get("subsets", None)
        if subsets is None:
            subsets = OOD_CV_SUBSETS.list()
        categories = config.get("categories", None)
        if categories is None:
            categories = OOD_CV_CATEGORIES.list()
        if categories is not None:
            categories = [
                MAP_OD3D_CATEGORIES[category]
                if category not in OOD_CV_CATEGORIES.list()
                else category
                for category in categories
            ]
        path_raw = OD3D_Dataset.get_path_raw(config=config)
        path_meta = OD3D_Dataset.get_path_meta(config=config)
        path_pascal3d_raw = Path(config.path_pascal3d_raw)
        rpath_meshes = Path("CAD")

        if config.extract_meta.remove_previous:
            logger.info("removing previous metas")
            if path_meta.exists():
                shutil.rmtree(path_meta)

        frames_names = []
        frames_categories = []
        frames_subsets = []
        for subset in subsets:
            for category in categories:
                # fpath_frame_names_partial = path_raw.joinpath("Image_sets", f"{category}_imagenet_{subset}.txt")
                fpath_frame_names_partial = path_raw.joinpath(
                    "lists",
                    f"{category}_{subset}.txt",
                )

                with fpath_frame_names_partial.open() as f:
                    frame_names_partial = f.read().splitlines()
                frames_names += [
                    Path(file_name.split(" ")[0]) for file_name in frame_names_partial
                ]
                frames_categories += [category] * len(frame_names_partial)
                frames_subsets += [subset] * len(frame_names_partial)

        # frames_subsets, frames_categories, frames_names = Pascal3DFrameMeta.get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw=path_raw, subsets=subsets, categories=categories)

        if config.get("frames", None) is not None:
            frames_names = list(filter(lambda f: f in config.frames, frames_names))

        for i in tqdm(range(len(frames_names))):
            fpath = path_meta.joinpath(
                OOD_CV_FrameMeta.get_rfpath_from_name_unique(
                    name_unique=OOD_CV_FrameMeta.get_name_unique_from_category_subset_name(
                        subset=frames_subsets[i],
                        category=frames_categories[i],
                        name=frames_names[i],
                    ),
                ),
            )
            if not fpath.exists() or config.extract_meta.override:
                subset = frames_subsets[i]
                category = frames_categories[i]
                rfpath_rgb = Path("images", category, f"{frames_names[i]}.JPEG")
                rfpath_annotation = Path(
                    "annotations",
                    category,
                    f"{frames_names[i]}.mat",
                )
                import scipy.io

                annotation = scipy.io.loadmat(path_raw.joinpath(rfpath_annotation))
                frame_meta = OOD_CV_FrameMeta.load_from_raw_annotation(
                    annotation=annotation,
                    rfpath_rgb=rfpath_rgb,
                    rpath_meshes=rpath_meshes,
                    category=category,
                    subset=subset,
                    path_raw=path_raw,
                    path_pascal3d_raw=path_pascal3d_raw,
                )

                """
                pascal3d_frame_meta = OOD_CV_FrameMeta.load_from_raw(frame_name=frames_names[i], subset='val',
                                                                      category=frames_categories[i],
                                                                      path_raw=path_pascal3d_raw, rpath_meshes=rpath_meshes)


                frame_meta = Pascal3D_OccFrameMeta(name=pascal3d_frame_meta.name, l_size=l_size,
                                                   rfpath_rgb=rfpath_rgb,
                                                   l_cam_intr4x4=pascal3d_frame_meta.l_cam_intr4x4,
                                                   l_cam_tform4x4_obj=pascal3d_frame_meta.l_cam_tform4x4_obj,
                                                   category=pascal3d_frame_meta.category,
                                                   rfpath_mesh=pascal3d_frame_meta.rfpath_mesh, subset=frames_subsets[i],
                                                   l_kpts3d=pascal3d_frame_meta.l_kpts3d,
                                                   l_kpts2d_annot=pascal3d_frame_meta.l_kpts2d_annot,
                                                   l_kpts2d_annot_vsbl=pascal3d_frame_meta.l_kpts2d_annot_vsbl,
                                                   kpts_names=pascal3d_frame_meta.kpts_names, rfpath_mask=rfpath_mask,
                                                   l_bbox=l_bbox)
                """

                if frame_meta is not None:
                    frame_meta.save(path_meta=path_meta)

        ##### PREPROCESS

    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == "cuboid" and config_preprocess.cuboid.get("enabled", False):
                override = config_preprocess.cuboid.get("override", False)
                remove_previous = config_preprocess.cuboid.get("remove_previous", False)
                self.preprocess_cuboid(
                    override=override,
                    remove_previous=remove_previous,
                )
            elif key == "mask" and config_preprocess.mask.get("enabled", False):
                override = config_preprocess.mask.get("override", False)
                remove_previous = config_preprocess.mask.get("remove_previous", False)
                self.preprocess_mask(override=override, remove_previous=remove_previous)
            elif key == "depth" and config_preprocess.depth.get("enabled", False):
                override = config_preprocess.depth.get("override", False)
                remove_previous = config_preprocess.depth.get("remove_previous", False)
                self.preprocess_depth(
                    override=override,
                    remove_previous=remove_previous,
                )

    def preprocess_cuboid(self, override=False, remove_previous=False):
        logger.info("preprocess cuboid...")

        for category in self.categories:
            if category not in self.all_categories:
                continue
            mesh_types = [
                OD3D_MESH_TYPES.CUBOID250,
                OD3D_MESH_TYPES.CUBOID500,
                OD3D_MESH_TYPES.CUBOID1000,
            ]
            for mesh_type in mesh_types:
                fpath_mesh_out = self.path_preprocess.joinpath(
                    OOD_CV_Frame.get_rfpath_pp_categorical_mesh(
                        mesh_type=mesh_type,
                        category=category,
                    ),
                )

                if fpath_mesh_out.exists() and not override:
                    logger.warning(f"mesh already exists {fpath_mesh_out}")
                    return
                else:
                    logger.info(
                        f"preprocessing mesh for {category} with type {mesh_type}",
                    )

                import re

                match = re.match(r"([a-z]+)([0-9]+)", mesh_type, re.I)
                if match and len(match.groups()) == 2:
                    mesh_type, mesh_vertices_count = match.groups()
                    mesh_vertices_count = int(mesh_vertices_count)
                else:
                    msg = f"could not retrieve mesh type and vertices count from mesh name {mesh_type}"
                    raise Exception(msg)

                fpaths_meshes_category = [
                    fpath
                    for fpath in self.path_pascal3d_raw.joinpath(
                        Pascal3DFrame.get_rpath_raw_categorical_meshes(
                            category=category,
                        ),
                    ).iterdir()
                ]
                meshes = Meshes.load_from_files(fpaths_meshes_category)
                meshes.verts.data = (
                    meshes.verts * PASCAL3D_SCALE_NORMALIZE_TO_REAL[category]
                )
                pts3d = meshes.verts

                from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d

                cuboids, _ = fit_cuboid_to_pts3d(
                    pts3d=pts3d,
                    optimize_rot=False,
                    optimize_transl=False,
                    vertices_max_count=mesh_vertices_count,
                    optimize_steps=1,
                )

                # show:
                # Meshes.load_from_meshes([meshes.get_mesh_with_id(i) for i in range(meshes.meshes_count)] + [cuboids.get_mesh_with_id(0)]).show(meshes_add_translation=False)

                obj_mesh = cuboids.get_mesh_with_id(0)
                obj_mesh.write_to_file(fpath=fpath_mesh_out)

    @property
    def path_meshes(self):
        return self.path_pascal3d_raw.joinpath("CAD")
