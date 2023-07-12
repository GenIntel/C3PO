import logging
logger = logging.getLogger(__name__)
import numpy as np
import torch.nn
import scipy.io
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
from pathlib import Path
import math
import od3d.io
from od3d.cv.geometry.mesh import Mesh
import pickle
from od3d.cv.geometry.transform import transf4x4_from_spherical
import shutil
from tqdm import tqdm
from od3d.datasets.dataset import OD3D_FRAME_MODALITIES
from od3d.cv.geometry.mesh import Meshes
from od3d.cv.geometry.primitives import Cuboids
from od3d.cv.io import save_ply

CATEGORIES = [
    "aeroplane",
    "bicycle",
    "boat",
    "bottle",
    "bus",
    "car",
    "chair",
    "diningtable",
    "motorbike",
    "sofa",
    "train",
    "tvmonitor",
]

SUBSETS = [
    "train",
    "val"
]
from typing import List
from dataclasses import dataclass
from od3d.datasets.dataset import OD3D_Frame
from omegaconf import OmegaConf
@dataclass
class Pascal3DFrame(OD3D_Frame):
    complete: bool
    kpts_names: List[str]
    l_bbox: List[float]
    l_kpts2d_annot: List[List[float]]
    l_kpts2d_annot_vsbl: List[bool]
    l_kpts3d: List[List[float]]
    incomplete_reason: str
    rfpath_mesh: Path
    path_meshes: Path
    _bbox = None
    _kpts2d_annot = None
    _kpts2d_annot_vsbl = None
    _kpts3d = None
    _mesh= None

    @staticmethod
    def load_from_raw(path_dataset: Path, path_preprocess: Path, rfpath_annotation: Path, rfpath_rgb: Path, path_meshes: Path, dt_shape_nemo=None, classes: list = None):
        annotation = scipy.io.loadmat(path_dataset.joinpath(rfpath_annotation))
        name = annotation['record']['filename'][0][0][0].split('.')[0]
        complete = True
        incomplete_reason = ""

        objects = annotation['record']['objects'][0][0][0]
        # assert len(objects) == 1
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

        rfpath_mesh = Path(category).joinpath(f"{(mesh_index + 1):02d}.off")

        fpath_mesh_kpoints3d = path_meshes.joinpath(f"{category}.mat")
        annotation_mesh3d = scipy.io.loadmat(fpath_mesh_kpoints3d)
        kpts3d = np.stack([annotation_mesh3d[category][n][0][mesh_index][0] if len(annotation_mesh3d[category][n][0][mesh_index]) > 0 else np.array([np.inf, np.inf, np.inf]) for n in kpts_names])
        kpts3d = torch.from_numpy(kpts3d)

        return Pascal3DFrame(name=name, complete=complete, incomplete_reason=incomplete_reason, path_dataset=path_dataset,
                      rfpath_rgb=rfpath_rgb, rfpath_mesh=rfpath_mesh, path_meshes=path_meshes, path_preprocess=path_preprocess,
                      l_bbox=bbox.tolist(), kpts_names=kpts_names, l_kpts2d_annot=kpts2d_annot.tolist(), l_kpts2d_annot_vsbl=kpts2d_annot_vsbl.tolist(), W=W, H=H, l_size=size.tolist(),
                      l_cam_tform4x4_obj=cam_tform4x4_obj.tolist(), l_cam_intr4x4=cam_intr4x4.tolist(), l_kpts3d=kpts3d.tolist(), category=category,
                             rfpath_mask=Path('mask').joinpath(f'{name}.png'), rfpath_depth=Path("None"), rfpath_depth_mask=Path("None")
                      )

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = torch.Tensor(self.l_bbox)
        return self._bbox


    @property
    def mask(self):
        if self._mask is None:
            from od3d.cv.io import read_image, save_image_mask
            from od3d.cv.geometry.mesh import Meshes
            fpath = self.path_preprocess.joinpath(self.rfpath_mask)
            if not fpath.exists():
                if torch.cuda.is_available():
                    device='cuda:0'
                else:
                    device = 'cpu'
                meshes = Meshes.load_from_files([self.path_meshes.joinpath(self.rfpath_mesh)], device=device)
                mask = meshes.render_feats(cams_tform4x4_obj=self.cam_tform4x4_obj[None,].to(device=device), cams_intr4x4=self.cam_intr4x4[None,].to(device=device),
                                           imgs_sizes=self.size.to(device=device), modality='mask')[0]
                save_image_mask(mask, path=fpath)
            self._mask = read_image(fpath) == 255
        return self._mask

    @property
    def kpts2d_annot(self):
        if self._kpts2d_annot is None:
            self._kpts2d_annot = torch.Tensor(self.l_kpts2d_annot)
        return self._kpts2d_annot

    @property
    def kpts2d_annot_vsbl(self):
        if self._kpts2d_annot_vsbl is None:
            self._kpts2d_annot_vsbl = torch.Tensor(self.l_kpts2d_annot_vsbl).to(dtype=bool)
        return self._kpts2d_annot_vsbl
    @property
    def kpts3d(self):
        if self._kpts3d is None:
            self._kpts3d = torch.Tensor(self.l_kpts3d)
        return self._kpts3d
    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = Mesh.load_from_file(fpath=self.path_meshes.joinpath(self.rfpath_mesh))
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


class Pascal3D(OD3D_Dataset):
    def __init__(
        self,
        config: DictConfig,
        transform=None
    ):
        super().__init__(config=config, transform=transform)

        if config.get("setup", False):
            Pascal3D.setup(config=self.config)
        if config.get("preprocess", False):
            Pascal3D.preprocess(config=self.config)


        self.path_meshes = self.path.joinpath("CAD")

        self.path_cuboids = self.config.path_cuboids

        self.subsets = self.config.get("subsets", SUBSETS)
        self.categories = self.config.get("classes", CATEGORIES)

        frames_names_meta = sorted([fpath.name.split('.')[0] for fpath in list(self.path_meta.joinpath("frames").iterdir())])

        self.frames_names, _ = Pascal3D.get_frame_names_from_subsets_and_cateogories(path_pascal3d_raw=self.path, subsets=self.subsets,categories=self.categories)
        self.frames_names = list(filter(lambda fn: fn in frames_names_meta, self.frames_names))

        self.name = config.name

        self.cache = self.config.cache
        if self.cache == "Disk":
            self.path_cache = Path(self.config.path_raw).parent.joinpath("PASCAL3D_CACHE")
            self.fpath_frame_names = self.path_cache.joinpath(self.name + '.txt')

            if self.fpath_frame_names.exists():
                with open(self.fpath_frame_names, 'r') as f:
                    self.frame_names = f.readlines()
            else:
                if not self.path_cache.exists():
                    self.path_cache.mkdir(parents=True)
                valid_frame_ids = []
                for i in range(len(self)):
                    frame = self.get_item_raw(item=i)
                    if frame is None:
                        continue
                    valid_frame_ids.append(i)
                    fpath_frame = self.path_cache.joinpath(frame.name + '.pkl')
                    if not fpath_frame.exists():
                        with open(fpath_frame, 'wb') as f:
                            pickle.dump(frame, f)
                self.frame_names = self.frame_names[valid_frame_ids]
                with open(self.fpath_frame_names, 'w') as f:
                    for frame_name in self.frame_names:
                        f.write(frame_name)

    @staticmethod
    def get_frame_names_from_subset_and_category(path_pascal3d_raw, subset, category):
        fpath_frame_names_partial = path_pascal3d_raw.joinpath("Image_sets", f"{category}_imagenet_{subset}.txt")
        with fpath_frame_names_partial.open() as f:
            frame_names_partial = f.read().splitlines()
            frame_rfpaths_partial = [f"{category}_imagenet/{name}" for name in frame_names_partial]
        return frame_names_partial, frame_rfpaths_partial

    @staticmethod
    def get_frame_names_from_subsets_and_cateogories(path_pascal3d_raw, subsets, categories):
        frames_rfpaths = []
        frames_names = []
        for subset in subsets:
            for category in categories:
                frame_names_partial, frame_rfpaths_partial = Pascal3D.get_frame_names_from_subset_and_category(path_pascal3d_raw=path_pascal3d_raw, subset=subset, category=category)
                frames_rfpaths += frame_rfpaths_partial
                frames_names += frame_names_partial

        return frames_names, frames_rfpaths
    @staticmethod
    def setup(config):
        path_pascal3d_raw = Path(config.path_raw)

        if path_pascal3d_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous Pascal3D+")
            shutil.rmtree(path_pascal3d_raw)

        if path_pascal3d_raw.exists():
            logger.info(f"Found Pascal3D+ dataset at {path_pascal3d_raw}")
        else:
            logger.info(f"Downloading Pascal3D+ dataset at {path_pascal3d_raw}")
            fpath = path_pascal3d_raw.joinpath("pascal3d.zip")
            od3d.io.download(url=config.url_pascal3d_raw, fpath=fpath)
            od3d.io.unzip(fpath=fpath, dst=fpath.parent)
            od3d.io.move_dir(src=fpath.parent.joinpath(Path(config.url_pascal3d_raw).with_suffix("").name),
                             dst=fpath.parent)

    @staticmethod
    def preprocess(config: DictConfig):
        Pascal3D.preprocess_cuboids(config=config)
        Pascal3D.preprocess_meta(config=config)

    @staticmethod
    def preprocess_cuboids(config: DictConfig):

        perc_axis_coverage = 0.99
        verts_count = 1000

        path = Path(config.path_raw)
        path_meshes = path.joinpath("CAD")
        path_preprocess = Path(config.path_preprocess)

        path_cuboids = path_preprocess.joinpath("cuboids")

        if not path_cuboids.exists() or config.preprocess_cuboids_override:
            for path_meshes_category in path_meshes.iterdir():
                if not path_meshes_category.is_dir():
                    continue
                paths_meshes_category = []
                for path_mesh_category in path_meshes_category.iterdir():
                    print(path_mesh_category)
                    paths_meshes_category.append(path_mesh_category)

                meshes = Meshes.load_from_files(paths_meshes_category)


                verts_count_axis_coverage = int(meshes.verts.shape[0] * perc_axis_coverage)

                verts_sorted = meshes.verts.sort(dim=0)[0]
                verts_group = verts_sorted[verts_count_axis_coverage::] - verts_sorted[0:-verts_count_axis_coverage]
                min_ids = verts_group.min(dim=0)[1]
                cuboid_limits = verts_sorted[torch.stack([min_ids, min_ids + verts_count_axis_coverage], dim=0)].diagonal(dim1=-2, dim2=-1)

                meshes = Cuboids.create_dense_from_limits(limits=cuboid_limits[None,], verts_count=verts_count)

                fpath = path_cuboids.joinpath(f'{path_meshes_category.name}.ply')
                fpath.parent.mkdir(parents=True, exist_ok=True)
                save_ply(fpath, verts=meshes.verts, faces=meshes.faces)


    @staticmethod
    def preprocess_meta(config:DictConfig):
        categories = config.get("classes", CATEGORIES)
        subsets = config.get("subsets", SUBSETS)

        path = Pascal3D.get_path(config=config)
        path_preprocess = Pascal3D.get_path_preprocess(config=config)
        path_meta = Pascal3D.get_path_meta(config=config)
        path_meshes = path.joinpath("CAD")

        # remove_previous = False, override = False

        if config.preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        frames_names, frames_rfpaths = Pascal3D.get_frame_names_from_subsets_and_cateogories(path_pascal3d_raw=path, subsets=subsets,categories=categories)

        if config.get('frames', None) is not None:
            frames_names = list(filter(lambda f: f in config.frames, frames_names))

        for i in tqdm(range(len(frames_names))):
            fpath = path_meta.joinpath("frames", frames_names[i] + '.yaml')
            if not config.preprocess_meta_override and fpath.exists():
                continue

            rfpath_annotation = Path("Annotations").joinpath(f"{frames_rfpaths[i]}.mat")
            rfpath_rgb = Path("Images").joinpath(f"{frames_rfpaths[i]}.JPEG")
            frame = Pascal3DFrame.load_from_raw(path_dataset=path, path_preprocess=path_preprocess, rfpath_rgb=rfpath_rgb, rfpath_annotation=rfpath_annotation, path_meshes=path_meshes)

            if frame.complete:
                conf = OmegaConf.structured(frame)
                if not fpath.parent.exists():
                    fpath.parent.mkdir(parents=True)
                OmegaConf.save(conf, fpath, resolve=True)
                _ = frame.mask

    #@staticmethod
    #def collate_fn(frames: List[Pascal3DFrame], modalities: List[OD3D_FRAME_MODALITIES]=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.MASK], device='cpu', dtype=torch.float32):
    #    frames = Pascal3DFrames(frames, modalities, dtype=dtype, device=device)
    #    return frames

    def __len__(self):
        return len(self.frames_names)

    def get_item(self, item):
        frame = self.get_item_raw(item)
        frame = self.transform(frame)
        frame.label = self.config.classes.index(frame.category)
        return frame

    def get_frame_by_name(self, frame_name):
        frame_config = OmegaConf.load(self.path_meta.joinpath('frames', frame_name + '.yaml'))
        return Pascal3DFrame(**frame_config)

    def get_frame_by_id(self, id):
        return self.get_frame_by_name(frame_name=self.frames_names[id])

    def get_item_raw(self, item):
        return self.get_frame_by_id(id=item)


from od3d.datasets.dataset import OD3D_Frames
class Pascal3DFrames(OD3D_Frames):
    def __init__(self, frames: List[Pascal3DFrame], modalities: List[OD3D_FRAME_MODALITIES], dtype, device):
        super().__init__(frames=frames, modalities=modalities, dtype=dtype, device=device)
        frame0 = frames[0]
        self.rfpaths_meshes = [frame.rfpath_mesh for frame in frames]
        self.path_meshes = frame0.path_meshes
