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
    path_co3d: Path
    rfpath_pcl: Path
    pcl_pts_count: int
    pcl_quality_score: float
    viewpoint_quality_score: float

    @staticmethod
    def load_from_raw(path_co3d: Path, sequence_annotation: SequenceAnnotation):
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

        return CO3D_Sequence(path_co3d=path_co3d, name=name, category=category, rfpath_pcl=rfpath_pcl,
                             pcl_pts_count=pcl_pts_count, pcl_quality_score=pcl_quality_score,
                             viewpoint_quality_score=viewpoint_quality_score)

    @property
    def pcl(self):
        if self._pcl is None:
            verts, _ = load_ply(str(self.path_co3d.joinpath(self.rfpath_pcl)))
            self._pcl = verts
        return self._pcl

@dataclass
class CO3D_Frame(OD3D_Frame):
    sequence_name: str
    frame_number: int
    depth_scale: float
    frame_type: str

    @property
    def depth(self):
        if self._depth is None:
            self._depth = read_co3d_depth_image(self.path_dataset.joinpath(self.rfpath_depth)) * self.depth_scale
        return self._depth

    @staticmethod
    def load_from_raw(path_co3d: Path, frame_annotation: FrameAnnotation):
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
        size = torch.Tensor([H, W]) #

        s = min(H, W)
        focal_length = torch.Tensor(frame_annotation.viewpoint.focal_length)  * s / 2.
        principal_point = -torch.Tensor(frame_annotation.viewpoint.principal_point) * s / 2. + size / 2.
        cam_intr4x4 = torch.Tensor([[focal_length[0], 0., principal_point[0], 0.],
                           [0., focal_length[1], principal_point[1], 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
        cam_proj4x4_obj = torch.bmm(cam_intr4x4[None,], cam_tform4x4_obj[None,])[0]

        l_size = size.tolist()
        l_cam_intr4x4 = cam_intr4x4.tolist()
        l_cam_tform4x4_obj = cam_tform4x4_obj.tolist()
        l_cam_proj4x4_obj = cam_proj4x4_obj.tolist()
        return CO3D_Frame(path_dataset=path_dataset, category=category, frame_number=frame_number, frame_type=frame_type,
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
        self.path = Path(config.path_co3d_raw)
        self.path_preprocess = Path(config.path_co3d_preprocess)
        self.path_meta = self.path_preprocess.joinpath('meta')

        self.device = "cpu"
        self.dtype = torch.float32
        self.whitelist_frame_types = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN,
                                      CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN,
                                      CO3D_FRAME_TYPES.TEST_KNOWN]

        self.classes = self.config.classes
        # [
        #    'car',
        #    #'carrot'
        #]

        self.preprocess_meta(remove_previous=self.config.preprocess_meta_remove_previous, override=self.config.preprocess_meta_override, sequences=self.config.get("sequences"))
        self.sequences_require_pcl = True
        self.frames_only_first_of_each_sequence = False

        logger.info("reading sequences meta...")

        self.sequences_names = list(sorted([fpath.name.split('.')[0] for fpath in self.path_meta.iterdir() if fpath.is_file()], key= lambda n: int(n)))
        if self.config.get('sequences', None) is not None:
            self.sequences_names = list(filter(lambda sequence_name: sequence_name in self.config.sequences, self.sequences_names))

        if self.sequences_require_pcl:
            self.sequences = []
            for sequence_name in self.sequences_names:
                sequence_config = OmegaConf.load( self.path_meta.joinpath(sequence_name + '.yaml'))
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

        if self.config.preprocess_pcls:
            self.preprocess_pcls()
        # sequence_names
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


    def preprocess_meta(self, remove_previous=False, override=False, sequences=None):
        if not override and self.path_meta.exists():
            return

        if remove_previous:
            if self.path_meta.exists():
                shutil.rmtree(self.path_meta)

        for cls in self.classes:
            logger.info(f'preprocess meta for class {cls}')
            sequence_annotations = load_dataclass_jgzip(
                f"{self.path}/{cls}/sequence_annotations.jgz", List[SequenceAnnotation]
            )


            logger.info('reading sequence annotations...')
            for sequence_annoation in tqdm(sequence_annotations):
                if sequences is not None and sequence_annoation.sequence_name not in sequences:
                    continue

                sequence = CO3D_Sequence.load_from_raw(path_co3d=self.path, sequence_annotation=sequence_annoation)
                config = OmegaConf.structured(sequence)
                fpath = self.path_meta.joinpath(sequence.name + '.yaml')
                if not fpath.parent.exists():
                    fpath.parent.mkdir(parents=True)
                OmegaConf.save(config, fpath, resolve=True)

            cls_frame_annotations = load_dataclass_jgzip(
                f"{self.path}/{cls}/frame_annotations.jgz", List[FrameAnnotation]
            )
            cls_frame_annotations = [fa for fa in cls_frame_annotations if fa.meta[
                'frame_type'] in self.whitelist_frame_types]

            logger.info('reading frame annotations...')
            for frame_annotation in tqdm(cls_frame_annotations):
                if sequences is not None and frame_annotation.sequence_name not in sequences:
                    continue
                frame = CO3D_Frame.load_from_raw(path_co3d=self.path, frame_annotation=frame_annotation)
                conf = OmegaConf.structured(frame)
                fpath = self.path_meta.joinpath(frame.sequence_name, frame.name + '.yaml')
                if not fpath.parent.exists():
                    fpath.parent.mkdir(parents=True)
                OmegaConf.save(conf, fpath, resolve=True)

    def preprocess_pcls(self):

        path_pcls = self.path_preprocess.joinpath('pcls')
        pts3d_max_count = 20000
        pts3d_prob_thresh = 0.6

        for seq_id in range(self.sequences_count):

            seq_dataset = torch.utils.data.Subset(dataset=self, indices=self.sequences_item_ids[seq_id])
            dataloader = torch.utils.data.DataLoader(dataset=seq_dataset, batch_size=10, shuffle=False,
                                                     collate_fn=partial(self.collate_fn, modalities=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.MASK]), num_workers=4)


            sequence = self.get_sequence_by_id(seq_id)
            pts3d = sequence.pcl

            pts3d = random_sampling(pts3d, pts3d_max_count=pts3d_max_count * 3)
            pts3d = voxel_downsampling(pts3d, K=pts3d_max_count)
            pts3d_prob = torch.ones(size=(pts3d.shape[0], 1), device=pts3d.device, dtype=pts3d.dtype)
            for frames in iter(dataloader):
                pxl2d = proj3d2d_broadcast(pts3d=pts3d, proj4x4=frames.cam_proj4x4_obj[:, None])
                pts3d_prob += sample_pxl2d_pts(frames.mask, pxl2d=pxl2d, padding_mode='zeros').sum(dim=0)

                #pxl2d = proj3d2d_broadcast(pts3d=pts3d_co3d, proj4x4=frame.cam_proj4x4_obj)
                #gb_with_mask_with_pts3d = draw_pixels(img=blend_rgb(frame.rgb, frame.mask*255.), pxls=pxl2d)
                #show_img(rgb_with_mask_with_pts3d)

            pts3d_prob = pts3d_prob / len(seq_dataset)
            # pts3d_co3ds = []
            # for prob in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:  #0.5, 0.6, 0.7, 0.8, 0.9, 0.95
            #    pts3d = torch.zeros_like(pts3d_co3d)
            #    pts3d[pts3d_co3d_prob[..., 0] > prob] = pts3d[pts3d_prob[..., 0] > prob]
            #    pts3d_co3ds.append(pts3d)
            # show_pcl(torch.stack(pts3d_co3ds, dim=0))
            # show_pcl(pts3d[pts3d_prob[..., 0] > 0.6])
            pts3d_clean = pts3d[pts3d_prob[..., 0] > pts3d_prob_thresh]
            #frame0: CO3D_Frame = seq_dataset[0]
            #show_img(frame0.rgb)
            #show_pcl(pts3d_clean, cam_tform4x4_obj=frame0.cam_tform4x4_obj, cam_intr4x4=frame0.cam_intr4x4, img_size=frame0.size)
            fpath_pcl = path_pcls.joinpath(sequence.name, f'co3d_probthresh_{str(pts3d_prob_thresh).replace(".", "_")}_max_{pts3d_max_count}' + '.ply')
            fpath_pcl.parent.mkdir(parents=True, exist_ok=True)
            save_ply(fpath_pcl, pts3d_clean)

            pca_tform_world = get_pca_tform_world(pts3d_clean)
            pca_pts3d_clean = transf3d_broadcast(pts3d_clean, pca_tform_world)

            cuboids_limits = torch.stack([pca_pts3d_clean.min(dim=-2)[0], pca_pts3d_clean.max(dim=-2)[0]], dim=-2)[None,]
            cuboid_pts3d_max_count = 1000
            cuboids = Cuboids(cuboids_limits=cuboids_limits, max_pts_count=cuboid_pts3d_max_count)
            cuboid_tform_pca = icp(cuboids.pts3d_surface[0], pca_pts3d_clean).inverse()
            cuboid_tform_world = cuboid_tform_pca[None,].bmm(pca_tform_world[None,])[0]
            fpath_pcl = path_pcls.joinpath(sequence.name, f'cuboid_max_{cuboid_pts3d_max_count}' + '.ply')

            faces = Meshes.get_faces_from_verts(verts=cuboids.pts3d_surface[0], ball_radius=1.)
            verts = transf3d_broadcast(pts3d=cuboids.pts3d_surface[0], transf4x4=cuboid_tform_world.inverse())

            save_ply(fpath_pcl, verts=verts, faces=faces)
            # show_pcl([pts3d_clean, transf3d_broadcast(pts3d=cuboids.pts3d_surface[0], transf4x4=cuboid_tform_world.inverse())])

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
