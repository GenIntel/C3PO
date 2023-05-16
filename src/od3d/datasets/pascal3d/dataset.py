import copy
import os
import logging
import BboxTools as bbt
import numpy as np
import pytorch3d.renderer
import torch.nn
import torchvision
from PIL import Image
import skimage
import scipy.io

from od3d.utils import construct_class_by_name
from od3d.utils import load_off

from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.pascal3d.setup import download_pascal3d, prepare_pascal3d
from omegaconf import DictConfig
from od3d.datasets.dtd import DTD
from pathlib import Path
import math
import PIL.Image
import cv2
import torchvision.io

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


class Pascal3DFrame:
    def __init__(self, fpath_annotation: Path, fpath_rgb: Path, path_meshes: Path):
        self.device = "cuda:1"
        self.dtype = torch.float32
        self.rgb = torchvision.io.read_image(str(fpath_rgb)).to(self.device)
        annotation = scipy.io.loadmat(fpath_annotation)

        self.name = annotation['record']['filename'][0][0][0].split('.')[0]

        objects = annotation['record']['objects'][0][0][0]
        # assert len(objects) == 1
        object = objects[0]
        self.category = object['class'][0]

        self.mesh_index = object['cad_index'][0][0] - 1
        self.bbox = torch.from_numpy(object['bbox'][0]).to(device=self.device, dtype=self.dtype)
        self.names_kpoints = list(object['anchors'][0][0].dtype.names)
        self.kpoints = np.stack([object['anchors'][0][0][n]['location'][0][0][0] if object['anchors'][0][0][n]['status'] == 1 else np.array([0, 0]) for n in self.names_kpoints])
        self.mask_kpoints_visible = np.array([True if object['anchors'][0][0][n]['status'] == 1 else False for n in self.names_kpoints])
        self.mask_kpoints_visible = torch.from_numpy(self.mask_kpoints_visible).to(device=self.device)
        self.width, self.height = self.rgb.shape[1:]
        self.size = torch.Tensor([self.width, self.height]).to(device=self.device, dtype=self.dtype)
        viewpoint = object['viewpoint']
        self.azimuth = viewpoint['azimuth'][0][0][0][0] * math.pi / 180
        self.elevation = viewpoint['elevation'][0][0][0][0] * math.pi / 180
        self.distance = viewpoint['distance'][0][0][0][0]
        focal = viewpoint['focal'][0][0][0][0]

        self.theta = viewpoint['theta'][0][0][0][0] * math.pi / 180
        principal = np.array([viewpoint['px'][0][0][0][0],
                              viewpoint['py'][0][0][0][0]])
        self.principal = torch.from_numpy(principal).to(device=self.device, dtype=self.dtype)
        viewport = viewpoint['viewport'][0][0][0][0]

        self.focal_length = torch.Tensor([viewpoint['focal'][0][0][0][0] * viewport,] ).to(device=self.device, dtype=self.dtype)
        self.principal_point = torch.Tensor([viewpoint['px'][0][0][0][0], viewpoint['py'][0][0][0][0]]).to(device=self.device, dtype=self.dtype)
        # self.K
        self.cam_tform_obj = self.calc_cam_tform_obj(azimuth=self.azimuth, elevation=self.elevation, theta=self.theta, distance=self.distance)
        cam_intr3x3 = np.array([[1. * viewport * focal, 0, principal[0]],
                           [0, 1. * viewport * focal, principal[1]],
                           [0, 0, 1.]])
        self.cam_intr4x4 = np.hstack((cam_intr3x3, [[0], [0], [0]]))
        self.cam_intr4x4 = np.vstack((self.cam_intr4x4, [0, 0, 0, 1]))
        self.cam_intr4x4 = torch.from_numpy(self.cam_intr4x4).to(device=self.device, dtype=self.dtype)

        self.cam_tform_obj = torch.Tensor(self.cam_tform_obj).to(self.device)
        self.cam_proj_obj = torch.matmul(self.cam_intr4x4, self.cam_tform_obj)

        self.fpath_mesh = path_meshes.joinpath(self.category, f"{(self.mesh_index + 1):02d}.off")

        fpath_mesh_kpoints3d = path_meshes.joinpath(f"{self.category}.mat")
        annotation_mesh3d = scipy.io.loadmat(fpath_mesh_kpoints3d)
        kpoints3d = np.stack([annotation_mesh3d[self.category][n][0][self.mesh_index][0] for n in self.names_kpoints])
        self.kpoints3d = torch.from_numpy(kpoints3d).to(device=self.device, dtype=self.dtype)
    def visualize(self):

        from od3d.cv.visual.draw import draw_pixels, draw_bbox
        from od3d.cv.geometry.transform import proj3d2d
        from od3d.cv.visual.render import render_mesh
        from od3d.cv.visual.blend import blend_rgb
        from od3d.cv.visual.show import show_img
        from od3d.cv.visual.crop import crop

        rgb_synthetic = render_mesh(fpath_mesh=self.fpath_mesh,
                                    cam_tform_obj=self.cam_tform_obj,
                                    cam_intr=self.cam_intr4x4, img_size=self.size)

        logging.info(f"Frame name {self.name}")



        #shift_uv = torch.Tensor([100, 100])
        #tform2d_sim = self.Tensor([scale, ])
        mix_real_with_synthetic = blend_rgb(self.rgb, rgb_synthetic)

        W_out = 400
        H_out = 200
        distance = 5.
        center = torch.LongTensor([500, 200]).to(device=self.device)
        center = torch.Tensor([(self.bbox[0] + self.bbox[2]) / 2., (self.bbox[1] + self.bbox[3]) / 2.]).to(
            device=self.device, dtype=self.dtype)
        center = proj3d2d(pts3d=torch.zeros(size=(1, 3)).to(device=self.device, dtype=self.dtype), proj4x4=self.cam_proj_obj)[0]

        scale = self.distance / distance
        mix_real_with_synthetic, cam_crop_tform_cam = crop(img=mix_real_with_synthetic, center=center, H_out=H_out, W_out=W_out, scale=scale)
        show_img(mix_real_with_synthetic)

        print(self.cam_proj_obj)
        cam_crop_proj_obj = torch.matmul(cam_crop_tform_cam, self.cam_proj_obj)
        pts2d_transf_proj = proj3d2d(pts3d=self.kpoints3d[self.mask_kpoints_visible], proj4x4=cam_crop_proj_obj)

        mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic, pts2d_transf_proj)
        show_img(mix_real_with_synthetic)


    def calc_cam_tform_obj(self, azimuth, elevation, theta, distance):
        if distance == 0:
            # return None
            distance = 0.1

        # camera center
        C = np.zeros((3, 1))
        C[0] = distance * math.cos(elevation) * math.sin(azimuth)
        C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
        C[2] = distance * math.sin(elevation)

        # rotate coordinate system by theta is equal to rotating the model by theta
        azimuth = -azimuth
        elevation = - (math.pi / 2 - elevation)

        # rotation matrix
        Rz = np.array([
            [math.cos(azimuth), -math.sin(azimuth), 0],
            [math.sin(azimuth), math.cos(azimuth), 0],
            [0, 0, 1],
        ])  # rotation by azimuth
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(elevation), -math.sin(elevation)],
            [0, math.sin(elevation), math.cos(elevation)],
        ])  # rotation by elevation

        R_rot = np.dot(Rx, Rz)
        R = np.hstack((R_rot, np.dot(-R_rot, C)))
        R = np.vstack((R, [0, 0, 0, 1]))

        R_theta = np.array(
            [[math.cos(self.theta), -math.sin(self.theta), 0, 0],
             [math.sin(self.theta), math.cos(self.theta), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])
        R = np.dot(R_theta, R)

        #T = R
        T = np.eye(4)
        T[0, :] = R[0, :]
        T[1, :] = -R[1, :]
        T[2, :] = -R[2, :]
        return T


class Pascal3D(OD3D_Dataset):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config=config)
        self.setup(self.config)

        self.path = Path(self.config.path_pascal3d_raw)
        self.path_meshes = self.path.joinpath("CAD")
        self.subsets = self.config.get("subsets", SUBSETS)
        self.categories = self.config.get("category", CATEGORIES)
        self.frame_names = []
        self.frame_rfpaths = []
        for subset in self.subsets:
            for category in self.categories:
                fpath_frame_names_partial = self.path.joinpath("Image_sets", f"{category}_imagenet_{subset}.txt")
                with fpath_frame_names_partial.open() as f:
                    frame_names_partial = f.read().splitlines()
                    frame_rfpaths_partial = [f"{category}_imagenet/{name}" for name in frame_names_partial]
                    self.frame_rfpaths += frame_rfpaths_partial
                    self.frame_names += frame_names_partial

    @staticmethod
    def setup(config):
        DTD.setup(config)
        download_pascal3d(config)
        # prepare_pascal3d(config)

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, item):
        item = item
        fpath_annotation = self.path.joinpath("Annotations", f"{self.frame_rfpaths[item]}.mat")
        fpath_rgb = self.path.joinpath("Images", f"{self.frame_rfpaths[item]}.JPEG")
        frame = Pascal3DFrame(fpath_annotation=fpath_annotation, fpath_rgb=fpath_rgb, path_meshes=self.path_meshes)
        return frame

    def visualize(self, item: int):
        frame: Pascal3DFrame = self.__getitem__(item=item)
        frame.visualize()
    def filter(self):
        if self.remove_no_bg is not None:
            filtered_file_list = []
            for i in range(len(self.file_list)):
                sample = self.__getitem__(i)
                obj_mask = skimage.measure.block_reduce(sample['obj_mask'], (self.remove_no_bg, self.remove_no_bg), np.max)
                if np.sum(1-obj_mask) >= 5:
                    filtered_file_list.append(self.file_list[i])
            self.file_list = filtered_file_list

        if self.segmentation_masks is not None:
            filtered_file_list = []
            for i in range(len(self.file_list)):
                sample = self.__getitem__(i)
                if 'inmodal' in self.segmentation_masks and len(sample['inmodal_mask'].shape) < 2:
                    continue
                if 'amodal' in self.segmentation_masks and len(sample['amodal_mask'].shape) < 2:
                    continue
                filtered_file_list.append(self.file_list[i])
            self.file_list = filtered_file_list


    def debug(self, item, save_dir=""):
        sample = self.__getitem__(item)
        img = sample["original_img"]
        kp, kpvis = sample["kp"], sample["kpvis"]
        y0, y1, x0, x1, _, _ = sample["bbox"]
        obj_mask = sample["obj_mask"]

        import cv2

        for i in range(len(kp)):
            if kpvis[i]:
                img = cv2.circle(
                    img, (int(kp[i, 1]), int(kp[i, 0])), 2, (255, 0, 0), -1
                )
        img = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        gray_img = (img * 0.3).astype(np.uint8)
        gray_img[obj_mask == 1] = img[obj_mask == 1]

        Image.fromarray(gray_img).save(
            os.path.join(save_dir, f'debug_{sample["this_name"].replace("/", "_")}.png')
        )
