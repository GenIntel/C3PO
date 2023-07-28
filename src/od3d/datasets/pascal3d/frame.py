import logging
logger = logging.getLogger(__name__)
from omegaconf import OmegaConf, DictConfig
import torch
from typing import List
from pathlib import Path
from dataclasses import dataclass
from od3d.datasets.frame import OD3D_FrameMeta, OD3D_Frame
import scipy.io
import math
import numpy as np
from od3d.cv.geometry.transform import transf4x4_from_spherical
from od3d.datasets.dataset import OD3D_Frames, OD3D_FRAME_MODALITIES
from od3d.datasets.pascal3d.enum import PASCAL3D_SCALE_NORMALIZE_TO_REAL, PASCAL3D_CATEGORIES
from od3d.cv.io import read_image, save_image_mask
from od3d.cv.geometry.mesh import Mesh, Meshes

from od3d.datasets.frame import OD3D_FrameMeta, \
    OD3D_FrameMetaMeshMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaRGBMixin, \
    OD3D_FrameMetaSizeMixin, OD3D_FrameKPTS2D3DMixin, OD3D_FrameMetaBBoxMixin, OD3D_FrameMetaSubsetMixin, \
    OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin

@dataclass
class Pascal3DFrameMeta(OD3D_FrameKPTS2D3DMixin, OD3D_FrameMetaBBoxMixin, OD3D_FrameMetaSubsetMixin,
                        OD3D_FrameMetaMeshMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaCamTform4x4ObjMixin,
                        OD3D_FrameMetaCamIntr4x4Mixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMeta):
    #complete: bool
    #incomplete_reason: str

    """
    @staticmethod
    def load_from_meta_with_rfpath(path_meta: Path, rfpath: Path):
        return Pascal3DFrameMeta(**Pascal3DFrameMeta.load_omega_conf_with_rfpath(path_meta=path_meta, rfpath=rfpath))
    """

    @property
    def name_unique(self):
        return f'{self.subset}/{self.category}/{self.name}'

    @staticmethod
    def get_name_unique_from_category_subset_name(category, subset, name):
        return f'{subset}/{category}/{name}'


    @staticmethod
    def load_from_raw(frame_name: str, subset: str, category: str, path_raw: Path, rpath_meshes: Path):
        frame_rfpath = f"{category}_imagenet/{frame_name}"
        rfpath_annotation = Path("Annotations").joinpath(f"{frame_rfpath}.mat")
        rfpath_rgb = Path("Images").joinpath(f"{frame_rfpath}.JPEG")

        annotation = scipy.io.loadmat(path_raw.joinpath(rfpath_annotation))
        name = annotation['record']['filename'][0][0][0].split('.')[0]

        objects = annotation['record']['objects'][0][0][0]
        # assert len(objects) == 1
        if len(objects) != 1:
            # complete = False
            incomplete_reason = f"num objects = {len(objects)}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        object = objects[0]
        category = object['class'][0]

        mesh_index = object['cad_index'][0][0] - 1
        # label = classes.index(meta.category)
        bbox = torch.from_numpy(object['bbox'][0])
        kpts_names = list(object['anchors'][0][0].dtype.names)
        kpts2d_annot = np.stack([object['anchors'][0][0][n]['location'][0][0][0] if object['anchors'][0][0][n]['status'] == 1 else np.array([0, 0]) for n in kpts_names])
        kpts2d_annot = torch.from_numpy(kpts2d_annot)
        kpts2d_annot_vsbl = np.array([True if object['anchors'][0][0][n]['status'] == 1 else False for n in kpts_names])
        kpts2d_annot_vsbl = torch.from_numpy(kpts2d_annot_vsbl)
        W = int(annotation['record'][0][0]['size']['width'][0][0][0][0])
        H = int(annotation['record'][0][0]['size']['height'][0][0][0][0]) # self.rgb.shape[1:]
        size = torch.Tensor([H, W])
        viewpoint = object['viewpoint']
        azimuth = viewpoint['azimuth'][0][0][0][0] * math.pi / 180
        elevation = viewpoint['elevation'][0][0][0][0] * math.pi / 180
        distance = viewpoint['distance'][0][0][0][0]
        focal = viewpoint['focal'][0][0][0][0]

        if focal == 0:
            complete = False
            incomplete_reason = "focal = 0"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
        theta = viewpoint['theta'][0][0][0][0] * math.pi / 180
        principal = np.array([viewpoint['px'][0][0][0][0],
                              viewpoint['py'][0][0][0][0]])
        viewport = viewpoint['viewport'][0][0][0][0]

        cam_tform4x4_obj = Pascal3DFrame.calc_cam_tform_obj(azimuth=azimuth, elevation=elevation, theta=theta, distance=distance)
        # cam_tform4x4_obj = torch.from_numpy(cam_tform4x4_obj)

        cam_intr3x3 = np.array([[1. * viewport * focal, 0, principal[0]],
                                [0, 1. * viewport * focal, principal[1]],
                                [0, 0, 1.]])
        cam_intr4x4 = np.hstack((cam_intr3x3, [[0], [0], [0]]))
        cam_intr4x4 = np.vstack((cam_intr4x4, [0, 0, 0, 1]))
        cam_intr4x4 = torch.from_numpy(cam_intr4x4).to(dtype=cam_tform4x4_obj.dtype)

        rfpath_mesh = path_raw.joinpath(rpath_meshes, category, f"{(mesh_index + 1):02d}.off")

        fpath_mesh_kpoints3d = path_raw.joinpath(rpath_meshes, f"{category}.mat")
        annotation_mesh3d = scipy.io.loadmat(fpath_mesh_kpoints3d)
        kpts3d = np.stack([annotation_mesh3d[category][n][0][mesh_index][0] if len(annotation_mesh3d[category][n][0][mesh_index]) > 0 else np.array([np.inf, np.inf, np.inf]) for n in kpts_names])
        kpts3d = torch.from_numpy(kpts3d)

        return Pascal3DFrameMeta(subset=subset, name=name, rfpath_rgb=rfpath_rgb, rfpath_mesh=rfpath_mesh,
                                 l_bbox=bbox.tolist(), kpts_names=kpts_names, l_kpts2d_annot=kpts2d_annot.tolist(),
                                 l_kpts2d_annot_vsbl=kpts2d_annot_vsbl.tolist(), l_size=size.tolist(),
                                 l_cam_tform4x4_obj=cam_tform4x4_obj.tolist(), l_cam_intr4x4=cam_intr4x4.tolist(),
                                 l_kpts3d=kpts3d.tolist(), category=category)

    """
    @staticmethod
    def get_dict_subset_category_frames_names(
            categories: List[PASCAL3D_CATEGORIES], path_meta: Path,
            dict_subset_category_frames_names=None):
        if dict_subset_category_frames_names is None:
            dict_subset_category_frames_names = {}
            # get subsets
            subsets = [fpath.stem for fpath in list(Pascal3DFrameMeta.get_path_frames(path_meta=path_meta).iterdir())]
            for subset in subsets:
                dict_subset_category_frames_names[subset] = None

        for subset, dict_category_frames_names in dict_subset_category_frames_names.items():
            if dict_category_frames_names is None:
                dict_category_frames_names = {category: None for category in categories}
                dict_subset_category_frames_names[subset] = dict_category_frames_names
            for category in categories:
                if dict_category_frames_names[category] is None:
                    frames_names = [fpath.stem for fpath in list(Pascal3DFrameMeta.
                                                                 get_path_frames_meta_with_subset_category(
                                                                        path_meta=path_meta, subset=subset,
                                                                        category=category).iterdir())]
                else:
                    frames_names = dict_category_frames_names[category]
                dict_subset_category_frames_names[subset][category] = frames_names
        return dict_subset_category_frames_names

    @staticmethod
    def get_dict_subset_category_frames_names_with_names_unique(self, names_unique: List[str]):
        dict_subset_category_frames_names = {}
        for name_unique in names_unique:
            subset, category, name = name_unique.split('/')
            if subset not in dict_subset_category_frames_names.keys():
                dict_subset_category_frames_names[subset] = {}
            if category not in dict_subset_category_frames_names[subset].keys():
                dict_subset_category_frames_names[subset][category] = []
            dict_subset_category_frames_names[subset][category].append(name)
        return dict_subset_category_frames_names


    @staticmethod
    def load_from_meta_with_name_unique(path_meta: Path, name_unique: str):
        subset, category, name = name_unique.split('/')
        rfpath = Pascal3DFrameMeta.get_rfpath_frame_meta_with_subset_category_name(subset=subset, category=category,
                                                                                   name=name)
        return Pascal3DFrameMeta.load_from_meta_with_rfpath(path_meta=path_meta, rfpath=rfpath)
    @staticmethod
    def load_from_meta_with_subset_category_name(path_meta: Path, subset: str, category: str, name: str):
        rfpath = Pascal3DFrameMeta.get_rfpath_frame_meta_with_subset_category_name(subset=subset, category=category,
                                                                                   name=name)
        return Pascal3DFrameMeta.load_from_meta_with_rfpath(path_meta=path_meta, rfpath=rfpath)


    def get_fpath(self, path_meta):
        return Pascal3DFrameMeta.get_fpath_frame_meta_with_category_name(path_meta=path_meta, subset=self.subset, category=self.category, name=self.name)

    @staticmethod
    def get_fpath_frame_meta_with_category_name(path_meta: Path, subset: str, category: str, name: str):
        return path_meta.joinpath(Pascal3DFrameMeta.get_rfpath_frame_meta_with_subset_category_name(subset=subset, category=category, name=name))

    @staticmethod
    def get_path_frames_meta_with_subset_category(path_meta: Path, subset: str, category: str):
        return path_meta.joinpath(Pascal3DFrameMeta.get_rpath_frames_meta_with_subset_category(subset=subset, category=category))

    @staticmethod
    def get_rfpath_frame_meta_with_subset_category_name(subset: str, category: str, name: str):
        return Pascal3DFrameMeta.get_rpath_frames_meta_with_subset_category(subset=subset, category=category).joinpath(name + '.yaml')

    @staticmethod
    def get_rpath_frames_meta_with_subset_category(subset: str, category: str):
        return Pascal3DFrameMeta.get_rfpath_frames().joinpath(subset).joinpath(category)


    @staticmethod
    def get_frames_names_from_subset_and_category_from_raw(path_pascal3d_raw, subset, category):
        fpath_frame_names_partial = path_pascal3d_raw.joinpath("Image_sets", f"{category}_imagenet_{subset}.txt")
        with fpath_frame_names_partial.open() as f:
            frame_names_partial = f.read().splitlines()
        return frame_names_partial

    @staticmethod
    def get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw, subsets, categories):
        frames_names = []
        frames_categories = []
        frames_subsets = []
        for subset in subsets:
            for category in categories:
                frame_names_partial = Pascal3DFrameMeta.get_frames_names_from_subset_and_category_from_raw(path_pascal3d_raw=path_pascal3d_raw, subset=subset, category=category)
                frames_names += frame_names_partial
                frames_categories += [category] * len(frame_names_partial)
                frames_subsets += [subset] * len(frame_names_partial)

        return frames_subsets, frames_categories, frames_names
        """

class Pascal3DFrame(OD3D_Frame):
    def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, path_meshes: Path, meta: Pascal3DFrameMeta, modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
        super().__init__(path_raw=path_raw, path_preprocess=path_preprocess, path_meta=path_meta, meta=meta, modalities=modalities, categories=categories)

        # , dt_shape_nemo=None, classes: list = None

    @property
    def cam_tform4x4_obj(self):
        if self._cam_tform4x4_obj is None:
            self._cam_tform4x4_obj = torch.Tensor(self.meta.l_cam_tform4x4_obj)
            self._cam_tform4x4_obj[2, 3] *= PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
        return self._cam_tform4x4_obj

    @property
    def fpath_mask(self):
        return self.path_preprocess.joinpath('mask', self.meta.name_unique + '.png')

    @property
    def mask(self):
        if self._mask is None:
            fpath = self.fpath_mask
            if not fpath.exists():
                self.preprocess_mask()
            self._mask = read_image(fpath) == 255
        return self._mask

    def preprocess_mask(self, override=False):
        if not self.fpath_mask.exists() or override:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
            meshes = Meshes.load_from_meshes([self.mesh], device=device)
            mask = meshes.render_feats(cams_tform4x4_obj=self.cam_tform4x4_obj[None,].to(device=device),
                                       cams_intr4x4=self.cam_intr4x4[None,].to(device=device),
                                       imgs_sizes=self.size.to(device=device), modality='mask')[0]
            save_image_mask(mask, path=self.fpath_mask)

    @property
    def kpts3d(self):
        if self._kpts3d is None:
            self._kpts3d = self.meta.kpts3d * PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
        return self._kpts3d

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = Mesh.load_from_file(fpath=self.fpath_mesh, scale=PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category])
        return self._mesh

    @staticmethod
    def calc_cam_tform_obj(azimuth, elevation, theta, distance):
        cam_tform4x4_obj = transf4x4_from_spherical(
            azim=torch.Tensor([azimuth]),
            elev=torch.Tensor([elevation]),
            theta=torch.Tensor([theta]),
            dist=torch.Tensor([distance]))[0]
        # cam_tform4x4_obj[0, :] = cam_tform4x4_obj[0, :]
        # cam_tform4x4_obj[1, :] = -cam_tform4x4_obj[1, :]
        # cam_tform4x4_obj[2, :] = -cam_tform4x4_obj[2, :]
        # cam_tform4x4_obj[2, 2:3] = -cam_tform4x4_obj[2, 2:3]
        return cam_tform4x4_obj


class Pascal3DFrames(OD3D_Frames):
    def __init__(self, frames: List[Pascal3DFrame], modalities: List[OD3D_FRAME_MODALITIES], dtype, device):
        super().__init__(frames=frames, modalities=modalities, dtype=dtype, device=device)
