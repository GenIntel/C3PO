import subprocess

from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_Frame
from omegaconf import DictConfig, OmegaConf
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
from od3d.cv.geometry.mesh import Meshes
from typing import List
import torchvision
import torch
from od3d.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3, transf3d_broadcast
from pathlib import Path
from od3d.cv.io import read_image, read_co3d_depth_image
from od3d.cv.visual.show import show_img
from od3d.cv.visual.blend import blend_rgb
import pytorch3d.transforms
import math
from enum import Enum
from od3d.cv.io import load_ply, save_ply
from od3d.io import run_cmd
from copy import copy

from od3d.cv.visual.show import show_pcl

from od3d.cv.geometry.transform import proj3d2d_broadcast
from od3d.cv.visual.sample import sample_pxl2d_pts
from od3d.cv.visual.draw import draw_pixels
from od3d.cv.visual.blend import blend_rgb

from od3d.cv.geometry.points_alignment import get_pca_tform_world
from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.cv.geometry.primitives import Cuboids
from od3d.cv.geometry.points_alignment import icp

import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
import shutil
from tqdm import tqdm
import torch.utils.data
from functools import partial
from od3d.cv.geometry.downsample import voxel_downsampling, random_sampling
from od3d.cv.transforms import RGB_UInt8ToFloat, RGB_Normalize, CenterZoom3D
import od3d.io
import cv2

class CO3D_FRAME_TYPES(str, Enum):
    DEV_KNOWN = 'dev_known'
    DEV_UNSEEN = 'dev_unseen'
    TEST_KNOWN = 'test_known'
    TEST_UNSEEN = 'test_unseen'
    TRAIN_KNOWN = 'train_known'
    TRAIN_UNSEEN = 'train_unseen'

class CO3D_FRAME_SPLITS(str, Enum):
    MULTISEQUENCE_CAR_DEV_KNOWN = 'multisequence_car_dev_known'
    MULTISEQUENCE_CAR_DEV_UNSEEN = 'multisequence_car_dev_unseen'
    MULTISEQUENCE_CAR_TEST_KNOWN = 'multisequence_car_test_known'
    MULTISEQUENCE_CAR_TEST_UNSEEN = 'multisequence_car_test_unseen'
    MULTISEQUENCE_CAR_TRAIN_KNOWN = 'multisequence_car_train_known'
    MULTISEQUENCE_CAR_TRAIN_UNSEEN = 'multisequence_car_train_unseen'
    SINGLESEQUENCE_CAR_TEST_0_KNOWN = 'singlesequence_car_test_0_known'
    SINGLESEQUENCE_CAR_TEST_0_UNSEEN = 'singlesequence_car_test_0_unseen'

class CO3D_CLASSES(str, Enum):
    APPLE = "apple"
    BACKPACK = "backpack"
    BALL = "ball"
    BANANA = "banana"
    BASEBALLBAT = "baseballbat"
    BASEBALLGLOVE = "baseballglove"
    BENCH = "bench"
    BICYCLE = "bicycle"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    BROCCOLI = "broccoli"
    CAKE = "cake"
    CAR = "car"
    CARROT = "carrot"
    CELLPHONE = "cellphone"
    CHAIR = "chair"
    COUCH = "couch"
    CUP = "cup"
    DONUT = "donut"
    FRISBEE = "frisbee"
    HAIRDRYER = "hairdryer"
    HANDBAG = "handbag"
    HOTDOG = "hotdog"
    HYDRANT = "hydrant"
    KEYBOARD = "keyboard"
    KITE = "kite"
    LAPTOP = "laptop"
    MICROWAVE = "microwave"
    MOTORCYCLE = "motorcycle"
    MOUSE = "mouse"
    ORANGE = "orange"
    PARKINGMETER = "parkingmeter"
    PIZZA = "pizza"
    PLANT = "plant"
    REMOTE = "remote"
    SANDWICH = "sandwich"
    SKATEBOARD = "skateboard"
    STOPSIGN = "stopsign"
    SUITCASE = "suitcase"
    TEDDYBEAR = "teddybear"
    TOASTER = "toaster"
    TOILET = "toilet"
    TOYBUS = "toybus"
    TOYPLANE = "toyplane"
    TOYTRAIN = "toytrain"
    TOYTRUCK = "toytruck"
    TV = "tv"
    UMBRELLA = "umbrella"
    VASE = "vase"
    WINEGLASS = "wineglass"

@dataclass
class CO3D_Sequence():
    name: str
    category: str
    _pcl = None
    _pcl_clean = None
    _front_name = None
    _cam_tform4x4_obj_canonic = None
    _cuboid = None
    path_co3d: Path
    path_preprocess: Path
    rfpath_pcl: Path
    pcl_pts_count: int
    pcl_quality_score: float
    viewpoint_quality_score: float
    path_meta: Path


    def preprocess_front_name(self, override=False):
        fpath_front_name = self.path_preprocess.joinpath('front_names', self.category, self.name, 'front_name.yaml')

        if override or not fpath_front_name.exists():

            config = OmegaConf.create()
            config.sequences = [self.name]
            config.path = []
            config.path_meta = self.path_meta
            config.classes = [self.category]
            dataset = CO3D(config=config)

            if fpath_front_name.exists():
                self._front_name = od3d.io.read_str_from_file(fpath_front_name)
                front_item_id = dataset.get_item_id_by_name(self.name, self._front_name)
            else:
                front_item_id = 0

            k = ord('a')  # 2424832
            while (k == ord('a') or k == ord('d')):  # k == 2424832 or k == 2555904:
                show_img(dataset.get_item(front_item_id).rgb, duration=1)
                k = cv2.waitKey(0)
                if k == ord('a'):  # 2424832: # :
                    # left key:
                    front_item_id -= 1
                elif k == ord('d'):  # 2555904:
                    # right key:
                    front_item_id += 1
                front_item_id %= len(dataset)
            self._front_name = dataset.get_item(front_item_id).name
            if not fpath_front_name.parent.exists():
                fpath_front_name.parent.mkdir(parents=True)
            od3d.io.write_str_to_file(fpath_front_name, text=self._front_name)

    def preprocess_cuboid(self, override=False):
        fpath_cuboid = self.path_preprocess.joinpath('cuboids', self.category, self.name + '.ply')
        if override and fpath_cuboid.exists():
            fpath_cuboid.unlink()
        _ = self.cuboid
    @staticmethod
    def load_from_raw(path_co3d: Path, path_meta: Path, path_preprocess: Path, sequence_annotation: SequenceAnnotation):
        name = sequence_annotation.sequence_name
        category = sequence_annotation.category
        if sequence_annotation.point_cloud is not None:
            rfpath_pcl = sequence_annotation.point_cloud.path
            pcl_pts_count = sequence_annotation.point_cloud.n_points
            pcl_quality_score = sequence_annotation.point_cloud.quality_score
        else:
            rfpath_pcl = Path('None')
            pcl_pts_count = 0
            pcl_quality_score = float('nan')

        viewpoint_quality_score = sequence_annotation.viewpoint_quality_score

        return CO3D_Sequence(path_co3d=path_co3d, path_meta=path_meta, path_preprocess=path_preprocess, name=name, category=category, rfpath_pcl=rfpath_pcl,
                             pcl_pts_count=pcl_pts_count, pcl_quality_score=pcl_quality_score,
                             viewpoint_quality_score=viewpoint_quality_score)

    @property
    def pcl(self):
        if self._pcl is None:
            fpath_pcl = self.path_co3d.joinpath(self.rfpath_pcl)
            verts, _ = load_ply(str(fpath_pcl))
            self._pcl = verts
        return self._pcl

    @property
    def front_name(self):
        fpath_front_name = self.path_preprocess.joinpath('front_names', self.category, self.name, 'front_name.yaml')
        if self._front_name is None:
            if not fpath_front_name.exists():
                self.preprocess_front_name()
            self._front_name = od3d.io.read_str_from_file(fpath_front_name)
        return self._front_name

    @property
    def fpath_cam_tform4x4_obj_canonic(self):
        return self.path_preprocess.joinpath('cam_tform4x4_obj_canonic', self.category, self.name, 'tform4x4.pt')

    def preprocess_cam_tform4x4_obj_canonic(self, override=False):
        fpath_cam_tform4x4_obj_canonic = self.fpath_cam_tform4x4_obj_canonic
        if override or not fpath_cam_tform4x4_obj_canonic.exists():
            front_frame_fpath_meta = self.path_meta.joinpath(self.name, self.front_name + '.yaml')
            frame_front_config = OmegaConf.load(front_frame_fpath_meta)
            frame_front = CO3D_Frame(**frame_front_config)
            pcl_clean = self.pcl_clean
            # frame_front.cam_tform4x4_obj # flip x and z axis for other direction


            # obj_canonic_tform4x4_obj = cam_tform_obj_canonic.inverse(), cam_tform_obj

            # pcl_clean = self.pcl_clean()
            cam_tform4x4_obj_canic = torch.Tensor([0., 0., 0.])
            torch.save(cam_tform4x4_obj_canic, f=str(fpath_cam_tform4x4_obj_canonic))
        pass
    @property
    def cam_tform4x4_obj_canonic(self):
        if self._cam_tform4x4_obj_canonic is None:
            if not self.fpath_cam_tform4x4_obj_canonic.exists():
                self.preprocess_cam_tform4x4_obj_canonic()
            self._cam_tform4x4_obj_canonic = torch.load(f=str(self.fpath_cam_tform4x4_obj_canonic))
        return self._cam_tform4x4_obj_canonic


    def preprocess_pcl_clean(self, override=False):
        fpath_pcl_clean = self.fpath_pcl_clean
        if override or not fpath_pcl_clean.exists():
            fpath_pcl_clean.parent.mkdir(parents=True, exist_ok=True)

            pts3d_max_count = 20000
            pts3d_prob_thresh = 0.6

            config = OmegaConf.create()
            config.sequences = [self.name]
            config.path = []
            config.path_meta = self.path_meta
            config.classes = [self.category]

            dataset = CO3D(config=config)
            dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False,
                                                     collate_fn=partial(CO3D.collate_fn,
                                                                        modalities=[OD3D_FRAME_MODALITIES.RGB,
                                                                                    OD3D_FRAME_MODALITIES.MASK]),
                                                     num_workers=4)

            pts3d = self.pcl

            pts3d = random_sampling(pts3d, pts3d_max_count=pts3d_max_count * 3)
            pts3d = voxel_downsampling(pts3d, K=pts3d_max_count)
            pts3d_prob = torch.ones(size=(pts3d.shape[0], 1), device=pts3d.device, dtype=pts3d.dtype)
            for frames in iter(dataloader):
                pxl2d = proj3d2d_broadcast(pts3d=pts3d, proj4x4=frames.cam_proj4x4_obj[:, None])
                pts3d_prob += sample_pxl2d_pts(frames.mask, pxl2d=pxl2d, padding_mode='zeros').sum(dim=0)

                # pxl2d = proj3d2d_broadcast(pts3d=pts3d_co3d, proj4x4=frame.cam_proj4x4_obj)
                # gb_with_mask_with_pts3d = draw_pixels(img=blend_rgb(frame.rgb, frame.mask*255.), pxls=pxl2d)
                # show_img(rgb_with_mask_with_pts3d)

            pts3d_prob = pts3d_prob / len(dataset)
            # pts3d_co3ds = []
            # for prob in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:  #0.5, 0.6, 0.7, 0.8, 0.9, 0.95
            #    pts3d = torch.zeros_like(pts3d_co3d)
            #    pts3d[pts3d_co3d_prob[..., 0] > prob] = pts3d[pts3d_prob[..., 0] > prob]
            #    pts3d_co3ds.append(pts3d)
            # show_pcl(torch.stack(pts3d_co3ds, dim=0))
            # show_pcl(pts3d[pts3d_prob[..., 0] > 0.6])
            pts3d_clean = pts3d[pts3d_prob[..., 0] > pts3d_prob_thresh]
            # frame0: CO3D_Frame = seq_dataset[0]
            # show_img(frame0.rgb)
            # show_pcl(pts3d_clean, cam_tform4x4_obj=frame0.cam_tform4x4_obj, cam_intr4x4=frame0.cam_intr4x4, img_size=frame0.size)

            save_ply(fpath_pcl_clean, pts3d_clean)

    @property
    def fpath_pcl_clean(self):
        return self.path_preprocess.joinpath('pcls', self.category, self.name, 'pcl_clean.ply') #  f'co3d_probthresh_{str(pts3d_prob_thresh).replace(".", "_")}_max_{pts3d_max_count}' + '.ply')
    @property
    def pcl_clean(self):
        if self._pcl_clean is None:
            fpath_pcl_clean = self.fpath_pcl_clean
            if not fpath_pcl_clean.exists():
                self.preprocess_pcl_clean()
            self._pcl_clean, _ = load_ply(fpath_pcl_clean)
        return self._pcl_clean

    @property
    def cuboid(self):
        if self._cuboid is None:

            fpath_cuboid = self.path_preprocess.joinpath('cuboids', self.category, self.name + '.ply')

            if not fpath_cuboid.exists():
                fpath_cuboid.parent.mkdir(parents=True, exist_ok=True)

                from od3d.cv.geometry.transform import se3_exp_map, tform4x4

                cuboid_pts3d_max_count = 1000

                pts3d_clean = self.pcl_clean

                pca_tform_world = get_pca_tform_world(pts3d_clean)
                pca_pts3d_clean = transf3d_broadcast(pts3d_clean, pca_tform_world)

                cuboids_limits = torch.stack([pca_pts3d_clean.min(dim=-2)[0], pca_pts3d_clean.max(dim=-2)[0]], dim=-2)[None,]

                cuboids = Cuboids.create_dense_from_limits(limits=cuboids_limits, verts_count=cuboid_pts3d_max_count)

                # verts, faces = cuboids.meshelize(number_vertices=cuboid_pts3d_max_count)

                icp_tform_pca = icp(cuboids.verts, pca_pts3d_clean).inverse()




                icp_tform_world = tform4x4(icp_tform_pca, pca_tform_world)

                icp_pts3d_clean = transf3d_broadcast(pts3d_clean, icp_tform_world)



                cuboid_tform6_tmp = torch.zeros(6).to(device=icp_pts3d_clean.device)
                tmp_tform4x4_icp = torch.eye(4).to(device=icp_pts3d_clean.device)

                for i in range(100):
                    tmp_tform4x4_icp = tform4x4(se3_exp_map(cuboid_tform6_tmp.detach()), tmp_tform4x4_icp)

                    cuboid_tform6_tmp = torch.nn.Parameter(torch.zeros(6).to(device=icp_pts3d_clean.device),
                                                           requires_grad=True)
                    optimizer = torch.optim.SGD(params=[cuboid_tform6_tmp], lr=0.0001)

                    cuboid_tform4x4_icp = tform4x4(se3_exp_map(cuboid_tform6_tmp), tmp_tform4x4_icp)

                    cuboid_pts3d = transf3d_broadcast(pts3d=icp_pts3d_clean, transf4x4=cuboid_tform4x4_icp)

                    _, icp_pts3d_ids_min = cuboid_pts3d.min(dim=0)
                    _, icp_pts3d_ids_max = cuboid_pts3d.max(dim=0)
                    cuboid_pts3d_limits = cuboid_pts3d[torch.cat([icp_pts3d_ids_min, icp_pts3d_ids_max], dim=0)]
                    icp_cuboids_vol = (cuboid_pts3d_limits[3, 0] - cuboid_pts3d_limits[0, 0]) * (cuboid_pts3d_limits[4, 1] - cuboid_pts3d_limits[1, 1]) * (cuboid_pts3d_limits[5, 2] - cuboid_pts3d_limits[2, 2])
                    loss = torch.norm(icp_cuboids_vol, p=2)
                    loss.backward()
                    logger.info(f'Volume {loss}')
                    optimizer.step()


                cuboids_limits = torch.stack([cuboid_pts3d.min(dim=-2)[0], cuboid_pts3d.max(dim=-2)[0]], dim=-2)[None,]
                cuboids = Cuboids.create_dense_from_limits(limits=cuboids_limits, verts_count=cuboid_pts3d_max_count)

                cuboid_tform_world = tform4x4(cuboid_tform4x4_icp, icp_tform_world) #  icp_tform_pca[None,].bmm(pca_tform_world[None,])[0]


                world_verts = transf3d_broadcast(pts3d=cuboids.verts, transf4x4=cuboid_tform_world.inverse())

                #show_pcl([pts3d_clean, world_verts])

                #faces = Meshes.get_faces_from_verts(verts=cuboids.pts3d_surface[0], ball_radius=1.)
                #verts = transf3d_broadcast(pts3d=cuboids.pts3d_surface[0], transf4x4=cuboid_tform_world.inverse())

                save_ply(fpath_cuboid, verts=world_verts, faces=cuboids.faces)


            self._cuboid = Cuboids.load_from_files(fpaths_meshes=[fpath_cuboid])
        return self._cuboid


@dataclass
class CO3D_Frame(OD3D_Frame):
    sequence_name: str
    path_meta: Path
    frame_number: int
    depth_scale: float
    frame_type: str
    _sequence = None
    @property
    def sequence(self):
        if self._sequence is None:
            sequence_config = OmegaConf.load(self.path_meta.joinpath(self.sequence_name + '.yaml'))
            self._sequence = CO3D_Sequence(**sequence_config)
        return self._sequence
    @property
    def depth(self):
        if self._depth is None:
            self._depth = read_co3d_depth_image(self.path_dataset.joinpath(self.rfpath_depth)) * self.depth_scale
        return self._depth

    @staticmethod
    def load_from_raw(path_co3d: Path, path_meta: Path, frame_annotation: FrameAnnotation):
        path_dataset = path_co3d
        category = frame_annotation.image.path.split('/')[0]
        sequence_name = frame_annotation.sequence_name
        frame_number = frame_annotation.frame_number
        frame_type = frame_annotation.meta['frame_type']
        name = f'{frame_annotation.frame_number}'

        rfpath_mask = Path(frame_annotation.mask.path)

        rfpath_rgb = Path(frame_annotation.image.path)

        depth_scale = frame_annotation.depth.scale_adjustment
        rfpath_depth = Path(frame_annotation.depth.path)

        rfpath_depth_mask = Path(frame_annotation.depth.mask_path)

        cam_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=torch.Tensor(frame_annotation.viewpoint.R).T, transl3=torch.Tensor(frame_annotation.viewpoint.T))
        default_tform_t3d = torch.Tensor([[-1., 0., 0., 0.],
                                         [0., -1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.]])
        cam_tform4x4_obj = torch.bmm(default_tform_t3d[None,], cam_tform4x4_obj[None,])[0]

        H, W = frame_annotation.image.size
        size = torch.Tensor([H, W])

        s = min(H, W)
        focal_length = torch.Tensor(frame_annotation.viewpoint.focal_length) * s / 2.
        principal_point = -torch.Tensor(frame_annotation.viewpoint.principal_point) * s / 2. + size.flip(dims=(0,)) / 2.
        cam_intr4x4 = torch.Tensor([[focal_length[0], 0., principal_point[0], 0.],
                           [0., focal_length[1], principal_point[1], 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
        cam_proj4x4_obj = torch.bmm(cam_intr4x4[None,], cam_tform4x4_obj[None,])[0]

        l_size = size.tolist()
        l_cam_intr4x4 = cam_intr4x4.tolist()
        l_cam_tform4x4_obj = cam_tform4x4_obj.tolist()
        l_cam_proj4x4_obj = cam_proj4x4_obj.tolist()
        return CO3D_Frame(path_dataset=path_dataset, path_meta=path_meta, category=category, frame_number=frame_number, frame_type=frame_type,
                   name=name, rfpath_mask=rfpath_mask, rfpath_depth=rfpath_depth, rfpath_depth_mask=rfpath_depth_mask,
                   rfpath_rgb=rfpath_rgb, H=H, W=W, l_size=l_size, l_cam_intr4x4=l_cam_intr4x4,
                   sequence_name=sequence_name,
                   l_cam_tform4x4_obj=l_cam_tform4x4_obj, l_cam_proj4x4_obj=l_cam_proj4x4_obj, depth_scale=depth_scale)

class CO3D(OD3D_Dataset):
    def __init__(
        self,
        config: DictConfig,
        transform=None
    ):
        super().__init__(config=config, transform=transform)

        if config.get("setup", False):
            CO3D.setup(config=self.config)
        if config.get("preprocess", False):
            CO3D.preprocess(config=self.config)

        self.path_meta = Path(config.path_meta)

        self.device = "cpu"
        self.dtype = torch.float32
        self.whitelist_frame_types = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN,
                                      CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN,
                                      CO3D_FRAME_TYPES.TEST_KNOWN]

        self.classes = self.config.get("classes", [])

        if self.config.get("fpaths_cuboids", None) is not None:
            self.cuboids = Meshes.load_from_files(fpaths_meshes=[self.config.fpaths_cuboids[cls] for cls in self.classes])

        # [
        #    'car',
        #    #'carrot'
        #]

        # CO3D.preprocess(config=self.config)

        # self.preprocess_meta(remove_previous=self.config.preprocess_meta_remove_previous, override=self.config.preprocess_meta_override, sequences=self.config.get("sequences"))
        self.sequences_require_pcl = True
        self.frames_only_first_of_each_sequence = False

        logger.info("reading sequences meta...")

        self.sequences_names = list(sorted([fpath.name.split('.')[0] for fpath in self.path_meta.iterdir() if fpath.is_file()], key= lambda n: int(n)))
        if self.config.get('sequences', None) is not None:
            self.sequences_names = list(filter(lambda sequence_name: sequence_name in self.config.sequences, self.sequences_names))

        if self.sequences_require_pcl:
            self.sequences = []
            for sequence_name in self.sequences_names:
                sequence_config = OmegaConf.load(self.path_meta.joinpath(sequence_name + '.yaml'))
                sequence = CO3D_Sequence(**sequence_config)
                self.sequences.append(sequence)

            # quality score lies in range [-2.35x, 1.04x]
            self.sequences = list(filter(lambda sequence: sequence.rfpath_pcl != Path('None') and sequence.pcl_quality_score > 0.8, self.sequences))
            self.sequences_names = [seq.name for seq in self.sequences]
        self.sequences_count = len(self.sequences_names)

        logger.info("reading frames names of sequences...")
        self.frames = []
        self.sequences_lengths = []
        self.frames_count = 0
        self.frames_names = []
        self.sequences_item_ids = []
        self.map_item_id_to_seq_id = []
        self.map_item_id_to_frame_id = []
        for sequence_name in tqdm(self.sequences_names):
            self.frames_names.append(sorted([fpath.name.split('.')[0] for fpath in list(self.path_meta.joinpath(sequence_name).iterdir())], key=lambda n: int(n)))
            self.sequences_lengths.append(len(self.frames_names[-1]))
            self.sequences_item_ids.append(list(range(self.frames_count, self.frames_count + self.sequences_lengths[-1])))
            self.frames_count += self.sequences_lengths[-1]
            self.map_item_id_to_frame_id += list(range(self.sequences_lengths[-1]))
            self.map_item_id_to_seq_id += list((len(self.sequences_lengths)-1, ) * self.sequences_lengths[-1])
            #self.sequences_map_name_to_id = {}
        #self.frames_map_name_to_id = {}

        # sequence_names



    @staticmethod
    def setup(config: DictConfig):

        # logger.info(OmegaConf.to_yaml(config))
        path_co3d_raw = Path(config.path_co3d_raw)
        if path_co3d_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous CO3D")
            shutil.rmtree(path_co3d_raw)

        if path_co3d_raw.exists() and not config.setup_override:
            logger.info(f"Found CO3D dataset at {path_co3d_raw}")
        else:
            path_co3d_repo = path_co3d_raw.joinpath('co3d')
            path_co3d_repo.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning CO3D github repository to {path_co3d_repo}")
            run_cmd(cmd=f'cd {path_co3d_raw} && git clone git@github.com:facebookresearch/co3d.git', live=True, logger=logger)
            logger.info(f"Downloading CO3D dataset at {path_co3d_raw}")
            run_cmd(cmd=f'python {path_co3d_repo.joinpath("co3d/download_dataset.py")} --download_folder {path_co3d_raw}', live=True, logger=logger)
            # --n_download_workers 1 --n_extract_workers 1

    def get_item_id_by_name(self, sequence_name, frame_name):
        for i in range(len(self)):
            seq_id = self.map_item_id_to_seq_id[i]
            frame_id = self.map_item_id_to_frame_id[i]
            if self.sequences_names[seq_id] == sequence_name and self.frames_names[seq_id][frame_id] == frame_name:
                return i
        return -1
    def get_sequence_by_name(self, sequence_name):
        sequence_config = OmegaConf.load(self.path_meta.joinpath(sequence_name + '.yaml'))
        return CO3D_Sequence(**sequence_config)

    def get_frame_by_name(self, sequence_name, frame_name):
        frame_config = OmegaConf.load(self.path_meta.joinpath(sequence_name, frame_name + '.yaml'))
        return CO3D_Frame(**frame_config)

    def get_frame_by_id(self, frame_id, seq_id):
        return self.get_frame_by_name(sequence_name=self.sequences_names[seq_id], frame_name=self.frames_names[seq_id][frame_id])

    def get_sequence_by_id(self, seq_id):
        return self.get_sequence_by_name(sequence_name=self.sequences_names[seq_id])

    @staticmethod
    def preprocess(config: DictConfig):
        logger.info("preprocess")
        CO3D.preprocess_meta(config=config)
        CO3D.preprocess_cuboids(config=config)
        # CO3D.preprocess_cam_tform4x4_obj_canonic(config=config)
        # CO3D.preprocess_front_names(config=config)

    @staticmethod
    def preprocess_meta(config: DictConfig):
        path = Path(config.path_co3d_raw)
        path_preprocess = Path(config.path_co3d_preprocess)
        path_meta = path_preprocess.joinpath('meta')
        sequences = config.get("sequences", None)
        whitelist_frame_types = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN,
                                  CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN,
                                  CO3D_FRAME_TYPES.TEST_KNOWN]

        if not config.preprocess_meta_override and path_meta.exists():
            return

        if config.preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        for cls in config.classes:
            logger.info(f'preprocess meta for class {cls}')
            sequence_annotations = load_dataclass_jgzip(
                f"{path}/{cls}/sequence_annotations.jgz", List[SequenceAnnotation]
            )
            logger.info('reading sequence annotations...')
            for sequence_annoation in tqdm(sequence_annotations):
                if sequences is not None and sequence_annoation.sequence_name not in sequences:
                    continue

                sequence = CO3D_Sequence.load_from_raw(path_co3d=path, path_meta=path_meta, path_preprocess=path_preprocess, sequence_annotation=sequence_annoation)
                config = OmegaConf.structured(sequence)
                fpath = path_meta.joinpath(sequence.name + '.yaml')
                if not fpath.parent.exists():
                    fpath.parent.mkdir(parents=True)
                OmegaConf.save(config, fpath, resolve=True)

            cls_frame_annotations = load_dataclass_jgzip(
                f"{path}/{cls}/frame_annotations.jgz", List[FrameAnnotation]
            )
            cls_frame_annotations = [fa for fa in cls_frame_annotations if fa.meta[
                'frame_type'] in whitelist_frame_types]

            logger.info('reading frame annotations...')
            for frame_annotation in tqdm(cls_frame_annotations):
                if sequences is not None and frame_annotation.sequence_name not in sequences:
                    continue
                frame = CO3D_Frame.load_from_raw(path_co3d=path, path_meta=path_meta, frame_annotation=frame_annotation)
                conf = OmegaConf.structured(frame)
                fpath = path_meta.joinpath(frame.sequence_name, frame.name + '.yaml')
                if not fpath.parent.exists():
                    fpath.parent.mkdir(parents=True)
                OmegaConf.save(conf, fpath, resolve=True)

    @staticmethod
    def preprocess_cuboids(config: DictConfig):
        logger.info("preprocess cuboids")
        config = copy(config)
        config.fpaths_cuboids = None
        config.setup = False
        config.preprocess = False
        dataset = CO3D(config=config)
        for sequence_name in dataset.sequences_names:
            logger.info(f"preprocess cuboids, sequence {sequence_name}")

            sequence = dataset.get_sequence_by_name(sequence_name=sequence_name)
            sequence.preprocess_cuboid(override=config.preprocess_cuboids_override)

    @staticmethod
    def preprocess_front_names(config: DictConfig):
        logger.info("preprocess front_names")
        config = copy(config)
        config.fpaths_cuboids = None
        config.setup = False
        config.preprocess = False
        dataset = CO3D(config=config)
        for sequence_name in dataset.sequences_names:
            logger.info(f"preprocess front_names, sequence {sequence_name}")
            sequence = dataset.get_sequence_by_name(sequence_name=sequence_name)
            sequence.preprocess_front_name(override=config.preprocess_front_names_override)

    @staticmethod
    def preprocess_cam_tform4x4_obj_canonic(config: DictConfig):
        logger.info("preprocess cam_tform4x4_obj_canonic")
        config = copy(config)
        config.fpaths_cuboids = None
        config.setup = False
        config.preprocess = False
        dataset = CO3D(config=config)
        for sequence_name in dataset.sequences_names:
            logger.info(f"preprocess cam_tform4x4_obj_canonic, sequence {sequence_name}")
            sequence = dataset.get_sequence_by_name(sequence_name=sequence_name)
            sequence.preprocess_cam_tform4x4_obj_canonic(override=config.preprocess_cam_tform4x4_obj_canonic_override)
    def __len__(self):
        return self.frames_count

    def get_item(self, item):
        seq_id = self.map_item_id_to_seq_id[item]
        frame_id = self.map_item_id_to_frame_id[item]
        frame = self.transform(self.get_frame_by_id(seq_id=seq_id, frame_id=frame_id))
        frame.label = self.config.classes.index(frame.category)
        return frame
    def visualize(self, item: int):
        pass
