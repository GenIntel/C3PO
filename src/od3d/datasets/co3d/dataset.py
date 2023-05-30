from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)
import torchvision
import torch
from od3d.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3, transf3d_broadcast
from pathlib import Path
from od3d.cv.io import read_image
from od3d.cv.visual.show import show_img
from od3d.cv.visual.blend import blend_rgb
import pytorch3d.transforms
import math
from enum import Enum
from od3d.cv.io import load_ply
from od3d.cv.visual.show import show_pcl

import logging
logger = logging.getLogger(__name__)

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

class CO3D_Frame():

    def __init__(self, path_co3d: Path, frame_annotation: FrameAnnotation, rfpath_pcl: Path, dtype, device):

        self.path_co3d = path_co3d
        self.complete = False
        self.device = device
        self.dtype = dtype
        self.category = frame_annotation.image.path.split('/')[0]
        self.sequence_name = frame_annotation.sequence_name
        self.frame_number = frame_annotation.frame_number
        self.name = f'{self.category}_{frame_annotation.sequence_name}_{frame_annotation.frame_number}'
        self.rgb = torchvision.io.read_image(str(self.path_co3d.joinpath(frame_annotation.image.path)), mode=torchvision.io.ImageReadMode.RGB).to(self.device)
        self.mask = torchvision.io.read_image(str(self.path_co3d.joinpath(frame_annotation.mask.path)), mode=torchvision.io.ImageReadMode.UNCHANGED).to(self.device)
        self.depth = read_image(self.path_co3d.joinpath(frame_annotation.depth.path)).to(self.device)

        self.rfpath_pcl = rfpath_pcl
        self._pcl = None

        self.cam_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=torch.Tensor(frame_annotation.viewpoint.R).T, transl3=torch.Tensor(frame_annotation.viewpoint.T)).to(self.device)
        default_tform_t3d = torch.Tensor([[-1., 0., 0., 0.],
                                         [0., -1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.]]).to(device=self.device, dtype=self.dtype)
        self.cam_tform4x4_obj = torch.bmm(default_tform_t3d[None,], self.cam_tform4x4_obj[None,])[0]



        self.H, self.W = self.mask.shape[1:]
        self.size = torch.Tensor([self.W, self.H]).to(dtype=self.dtype, device=self.device)
        s = min(self.H, self.W)
        focal_length = torch.Tensor(frame_annotation.viewpoint.focal_length).to(dtype=self.dtype, device=self.device) * s / 2.
        principal_point = -torch.Tensor(frame_annotation.viewpoint.principal_point).to(dtype=self.dtype, device=self.device) * s / 2. + self.size  / 2.
        self.cam_intr4x4 = torch.Tensor([[focal_length[0], 0., principal_point[0], 0.],
                           [0., focal_length[1], principal_point[1], 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]]).to(device=self.device, dtype=self.dtype)
        self.cam_proj4x4_obj = torch.bmm(self.cam_intr4x4[None,], self.cam_tform4x4_obj[None,])[0]

    @property
    def pcl(self):
        if self._pcl is None:
            np_verts, _ = load_ply(str(self.path_co3d.joinpath(self.rfpath_pcl)))
            self._pcl = np_verts.to(dtype=self.dtype, device=self.device)
        return self._pcl
    def to(self, device: torch.device):
        if self.device != device:
            for k, a in self.__dict__.items():
                if isinstance(a, torch.Tensor):
                    setattr(self, k, a.to(device))
                    # self.__dict__[k] = a.to(device)
            self.device = device
class CO3D_Frames():
    def __init__(self, frames: list[CO3D_Frame]):
        for frame in frames:
            frame.to('cuda:0')

        frame0 = frames[0]
        self.dtype = frame0.dtype
        self.device = frame0.device
        self.path_co3d = frame0.path_co3d
        self.size = frame0.size
        self.rgb = torch.stack([frame.rgb for frame in frames], dim=0)
        self.mask = torch.stack([frame.mask for frame in frames], dim=0)
        self.depth = torch.stack([frame.depth for frame in frames], dim=0)
        self.cam_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0)
        self.cam_proj4x4_obj = torch.stack([frame.cam_proj4x4_obj for frame in frames], dim=0)
        self.cam_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0)
        self.rfpath_pcl = [frame.rfpath_pcl for frame in frames]

    def visualize(self):

        # show_pcl
        print(self.rfpath_pcl[0])
        # show_img(self.rgb[0])

        verts, _ = load_ply(str(self.path_co3d.joinpath(self.rfpath_pcl[0])))
        verts = verts.to(self.device)

        show_pcl(verts, cam_tform4x4_obj=self.cam_tform4x4_obj[0], cam_intr4x4=self.cam_intr4x4[0], img_size=self.size)


        # verts, faces = load_ply(filename)
        mix_real_with_synthetic = blend_rgb(self.rgb[0], self.mask[0])

        #mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic,
        #                                      proj3d2d_broadcast(pts3d=torch.cat((pts3d, self.kpts3d[0, self.kpts3d_vsbl[0]])),
        #                                               proj4x4=self.cam_proj4x4_obj[0]), colors=(0, 255, 0))
        #mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic, self.kpts2d_annot[0, self.kpts2d_annot_vsbl[0]],
        #                                     colors=(0, 0, 255), radius_in=2, radius_out=4)

        show_img(mix_real_with_synthetic)
class CO3D(OD3D_Dataset):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config=config)
        self.path = Path(config.path_co3d_raw)

        self.frame_annotations = []
        self.sequence_annotations = []

        self.device = "cpu"
        self.dtype = torch.float32

        self.classes = [
            'car',
            #'carrot'
        ]
        self.sequences_require_pcl = True
        self.frames_only_first_of_each_sequence = True
        self.whitelist_frame_types = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN, CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN, CO3D_FRAME_TYPES.TEST_KNOWN]
        self.sequences_names = []
        self.sequences_map_names_to_id = {}
        self.sequences_rfpaths_pcl = []
        for cls in self.classes:
            sequence_annotations = load_dataclass_jgzip(
                f"{self.path}/{cls}/sequence_annotations.jgz", list[SequenceAnnotation]
            )
            if self.sequences_require_pcl:
                count_sequences_before_pcl_filter = len(sequence_annotations)
                sequence_annotations = list(filter(lambda sa: sa.point_cloud is not None, sequence_annotations))
                count_sequences_after_pcl_filter = len(sequence_annotations)
                logger.info(f'Keep {count_sequences_after_pcl_filter}/{count_sequences_before_pcl_filter} sequences with pointclouds for class {cls}')

            cls_sequences_names = [a.sequence_name for a in sequence_annotations]
            self.sequences_names += cls_sequences_names
            self.sequences_map_names_to_id.update({sequence_annotations[i].sequence_name: i + len(self.sequences_map_names_to_id) for i in range(len(sequence_annotations))})
            self.sequences_rfpaths_pcl += [a.point_cloud.path for a in sequence_annotations]
            cls_frame_annotations = load_dataclass_jgzip(
                f"{self.path}/{cls}/frame_annotations.jgz", list[FrameAnnotation]
            )
            cls_frame_annotations = [fa for fa in cls_frame_annotations if fa.meta['frame_type'] in self.whitelist_frame_types and fa.sequence_name in cls_sequences_names]
            if self.frames_only_first_of_each_sequence:
                cls_frame_annotations_filtered = []
                cls_frame_annotations_filtered_sequence_names = []
                for fa in cls_frame_annotations:
                    if fa.sequence_name not in cls_frame_annotations_filtered_sequence_names:
                        cls_frame_annotations_filtered.append(fa)
                        cls_frame_annotations_filtered_sequence_names.append(fa.sequence_name)
                cls_frame_annotations = cls_frame_annotations_filtered
            self.frame_annotations += cls_frame_annotations

            from scipy.linalg import orthogonal_procrustes
            from od3d.cv.geometry.transform import pts3d_to_pts4d

            frame1_pts3d_1 = self.__getitem__(0).pcl
            frame2_pts3d_2 = self.__getitem__(1).pcl
            #frame1_pts4d_1 = pts3d_to_pts4d(pts3d=frame1_pts3d_1)
            #frame2_pts4d_2 = pts3d_to_pts4d(pts3d=frame2_pts3d_2)
            #frame2_tform_frame1, res = orthogonal_procrustes(frame1_pts4d_1, frame2_pts4d_2, check_finite=False)
            #frame2_tform_frame1 = torch.from_numpy(frame2_tform_frame1).to(device=self.device, dtype=self.dtype)
            from pytorch3d.ops.points_alignment import iterative_closest_point
            icp_sol = iterative_closest_point(X=frame1_pts3d_1[None,].cuda(), Y=frame2_pts3d_2[None,].cuda(), estimate_scale=True, verbose=False, max_iterations=20)
            #icp_sol.RTs
            #icp_sol.Xt
            # frame2_tform_pts3d_1 = transf3d_broadcast(pts3d=frame1_pts3d_1, transf4x4=frame2_tform_frame1.inverse())
            frame2_tform_frame1 = transf4x4_from_rot3x3_and_transl3(rot3x3=icp_sol.RTs.R.transpose(-1, -2), transl3=icp_sol.RTs.T)[0].cpu()
            frame2_tform_pts3d_1 = transf3d_broadcast(pts3d=frame1_pts3d_1, transf4x4=frame2_tform_frame1)
            show_pcl(torch.stack([frame1_pts3d_1, frame2_pts3d_2, frame2_tform_pts3d_1], dim=0))
    def __len__(self):
        return len(self.frame_annotations)

    def __getitem__(self, item):
        return CO3D_Frame(path_co3d=self.path, frame_annotation=self.frame_annotations[item], rfpath_pcl=self.sequences_rfpaths_pcl[self.sequences_map_names_to_id[self.frame_annotations[item].sequence_name]], dtype=self.dtype, device=self.device)

    def visualize(self, item: int):
        pass
    @staticmethod
    def collate_fn(bla):
        frames = CO3D_Frames(bla)
        return frames