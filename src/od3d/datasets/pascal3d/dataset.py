import copy
import os
import logging
logger = logging.getLogger(__name__)
import BboxTools as bbt
import numpy as np
import pytorch3d.renderer
import torch.nn
import torchvision
from PIL import Image
import skimage
import scipy.io
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.pascal3d.setup import download_pascal3d, prepare_pascal3d
from omegaconf import DictConfig
from od3d.datasets.dtd import DTD
from pathlib import Path
import math
import torchvision.io
from od3d.cv.visual.draw import draw_pixels, draw_bbox
from od3d.cv.geometry.transform import proj3d2d, reproj2d3d
from od3d.cv.geometry.transform import transf3d
from od3d.cv.visual.render import render_mask, render_depth
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.visual.show import show_img
from od3d.cv.visual.crop import crop
from od3d.cv.visual.sample import sample_pxl2d_pts

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
    def __init__(self, fpath_annotation: Path, fpath_rgb: Path, path_meshes: Path, fpath_txtr: Path = None):
        self.device = "cpu"
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
        self.kpts_names = list(object['anchors'][0][0].dtype.names)
        kpts2d_annot = np.stack([object['anchors'][0][0][n]['location'][0][0][0] if object['anchors'][0][0][n]['status'] == 1 else np.array([0, 0]) for n in self.kpts_names])
        self.kpts2d_annot = torch.from_numpy(kpts2d_annot).to(device=self.device, dtype=self.dtype)
        kpts2d_annot_vsbl = np.array([True if object['anchors'][0][0][n]['status'] == 1 else False for n in self.kpts_names])
        self.kpts2d_annot_vsbl = torch.from_numpy(kpts2d_annot_vsbl).to(device=self.device)
        width, height = self.rgb.shape[1:]
        self.size = torch.Tensor([width, height]).to(device=self.device, dtype=self.dtype)
        viewpoint = object['viewpoint']
        azimuth = viewpoint['azimuth'][0][0][0][0] * math.pi / 180
        elevation = viewpoint['elevation'][0][0][0][0] * math.pi / 180
        distance = viewpoint['distance'][0][0][0][0]
        focal = viewpoint['focal'][0][0][0][0]

        theta = viewpoint['theta'][0][0][0][0] * math.pi / 180
        principal = np.array([viewpoint['px'][0][0][0][0],
                              viewpoint['py'][0][0][0][0]])
        viewport = viewpoint['viewport'][0][0][0][0]

        cam_tform4x4_obj = self.calc_cam_tform_obj(azimuth=azimuth, elevation=elevation, theta=theta, distance=distance)
        self.cam_tform4x4_obj = torch.Tensor(cam_tform4x4_obj).to(self.device)

        cam_intr3x3 = np.array([[1. * viewport * focal, 0, principal[0]],
                           [0, 1. * viewport * focal, principal[1]],
                           [0, 0, 1.]])
        cam_intr4x4 = np.hstack((cam_intr3x3, [[0], [0], [0]]))
        cam_intr4x4 = np.vstack((cam_intr4x4, [0, 0, 0, 1]))
        self.cam_intr4x4 = torch.from_numpy(cam_intr4x4).to(device=self.device, dtype=self.dtype)

        self.cam_proj4x4_obj = torch.matmul(self.cam_intr4x4, self.cam_tform4x4_obj)

        if fpath_txtr is not None:
            self.txtr = torchvision.io.read_image(str(fpath_txtr)).to(self.device)
        else:
            self.txtr = None

        self.fpath_mesh = path_meshes.joinpath(self.category, f"{(self.mesh_index + 1):02d}.off")
        self.mask = render_mask(fpath_mesh=self.fpath_mesh,
                                cam_tform_obj=self.cam_tform4x4_obj.to("cuda:0"),
                                cam_intr=self.cam_intr4x4.to("cuda:0"), img_size=self.size.to("cuda:0")).to(self.device)

        self.depth = render_depth(fpath_mesh=self.fpath_mesh,
                                 cam_tform_obj=self.cam_tform4x4_obj.to("cuda:0"),
                                 cam_intr=self.cam_intr4x4.to("cuda:0"), img_size=self.size.to("cuda:0")).to(self.device)

        fpath_mesh_kpoints3d = path_meshes.joinpath(f"{self.category}.mat")
        annotation_mesh3d = scipy.io.loadmat(fpath_mesh_kpoints3d)
        kpts3d = np.stack([annotation_mesh3d[self.category][n][0][self.mesh_index][0] for n in self.kpts_names])
        self.kpts3d = torch.from_numpy(kpts3d).to(device=self.device, dtype=self.dtype)
        self.kpts2d = proj3d2d(pts3d=self.kpts3d, proj4x4=self.cam_proj4x4_obj)
        #self.kpts3d_vsbl =
        cam_tform_kpts3d_depth_rendered = sample_pxl2d_pts(self.depth, self.kpts2d)[:, 0]
        cam_tform_kpts3d_depth = transf3d(self.kpts3d, transf4x4=self.cam_tform4x4_obj)[:, 2]
        self.kpts3d_vsbl = ( cam_tform_kpts3d_depth - 0.2 < cam_tform_kpts3d_depth_rendered) + (cam_tform_kpts3d_depth_rendered <= 0.)

    def augment(self, H, W, dist, txtr):
        logger.info(f"Frame name {self.name}")

        #center = torch.LongTensor([500, 200]).to(device=self.device)
        #center = torch.Tensor([(self.bbox[0] + self.bbox[2]) / 2., (self.bbox[1] + self.bbox[3]) / 2.]).to(
        #    device=self.device, dtype=self.dtype)
        center = proj3d2d(pts3d=torch.zeros(size=(1, 3)).to(device=self.device, dtype=self.dtype), proj4x4=self.cam_proj4x4_obj)[0]
        scale = self.cam_tform4x4_obj[2, 3] / dist
        self.cam_tform4x4_obj[2, 3] = self.cam_tform4x4_obj[2, 3] / scale
        self.cam_intr4x4[:2, 2] = self.cam_intr4x4[:2, 2] * scale
        self.kpts2d_annot = self.kpts2d_annot * scale
        self.bbox = self.bbox * scale

        self.cam_proj4x4_obj = torch.matmul(self.cam_intr4x4, self.cam_tform4x4_obj)

        #mix_real_with_synthetic, cam_crop_tform_cam = crop(img=mix_real_with_synthetic, center=center, H_out=H_out, W_out=W_out, scale=scale, ctx=self.txtr)

        self.rgb, cam_crop_tform_cam = crop(img=self.rgb, center=center, H_out=H, W_out=W, scale=scale, ctx=self.txtr)
        self.mask, cam_crop_tform_cam = crop(img=self.mask, center=center, H_out=H, W_out=W, scale=scale, ctx=self.txtr)

        # we already account for the scale with the transformation, but we cannot do that for the padding
        cam_crop_tform_cam[0, 0] = 1.
        cam_crop_tform_cam[1, 1] = 1.
        self.cam_intr4x4 = torch.matmul(cam_crop_tform_cam, self.cam_intr4x4)
        self.cam_proj4x4_obj = torch.matmul(self.cam_intr4x4, self.cam_tform4x4_obj)

        self.kpts2d = proj3d2d(pts3d=self.kpts3d, proj4x4=self.cam_proj4x4_obj)
        self.kpts2d_annot = self.kpts2d_annot + cam_crop_tform_cam[:2, 2]
        self.bbox[[0, 2]] = self.bbox[[0, 2]] + cam_crop_tform_cam[0, 2]
        self.bbox[[1, 3]] = self.bbox[[1, 3]] + cam_crop_tform_cam[1, 2]

        pass
    def visualize(self):
        #this_size = cfg.image_sizes[cate]
        #out_shape = [
        #    ((this_size[0] - 1) // 32 + 1) * 32,
        #    ((this_size[1] - 1) // 32 + 1) * 32,
        #]
        #out_shape = [int(out_shape[0]), int(out_shape[1])]

        W_out = 400
        H_out = 200
        dist = 10.
        pts3d = torch.zeros(size=(1, 3)).to(device=self.device, dtype=self.dtype)

        # self.augment(H=H_out, W=W_out, dist=10, txtr=self.txtr)
        #logging.info(self.cam_tform4x4_obj.inverse()[:3, 3].norm())

        mix_real_with_synthetic = blend_rgb(self.rgb, self.mask)

        mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic, proj3d2d(pts3d=torch.cat((pts3d, self.kpts3d[self.kpts3d_vsbl])), proj4x4=self.cam_proj4x4_obj), colors=(0, 255, 0))
        mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic, self.kpts2d_annot[self.kpts2d_annot_vsbl], colors=(0, 0, 255), radius_in=2, radius_out=4)

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
            [[math.cos(theta), -math.sin(theta), 0, 0],
             [math.sin(theta), math.cos(theta), 0, 0],
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

        if config.pad_texture:
            self.txtrs_path = Path(config.path_dtd_raw)
            self.txtrs_rfpaths = []
            self.txtrs_rfpaths_choices_per_subset = {"train": [], "val": []}
            txtrs_meta = scipy.io.loadmat(self.txtrs_path.joinpath("imdb", "imdb.mat"))["images"]
            txtrs_ids = txtrs_meta[0, 0][0][0]
            txtrs_fnames = txtrs_meta[0, 0][1][0]
            txtrs_fnames = np.array([f[0] for f in txtrs_fnames])
            txtrs_subsets = txtrs_meta[0, 0][2][0]
            for j, f, s in zip(txtrs_ids, txtrs_fnames, txtrs_subsets):
                if s == 1:
                    self.txtrs_rfpaths_choices_per_subset["train"].append(f)
                elif s == 2:
                    self.txtrs_rfpaths_choices_per_subset["val"].append(f)

        for subset in self.subsets:
            for category in self.categories:
                fpath_frame_names_partial = self.path.joinpath("Image_sets", f"{category}_imagenet_{subset}.txt")
                with fpath_frame_names_partial.open() as f:
                    frame_names_partial = f.read().splitlines()
                    frame_rfpaths_partial = [f"{category}_imagenet/{name}" for name in frame_names_partial]
                    self.frame_rfpaths += frame_rfpaths_partial
                    self.frame_names += frame_names_partial

                    if config.pad_texture:
                        txtrs_rfpaths_partial = [f"images/{np.random.choice(self.txtrs_rfpaths_choices_per_subset[subset])}" for n in frame_names_partial]
                        self.txtrs_rfpaths += txtrs_rfpaths_partial

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
        if self.config.pad_texture:
            fpath_txtr = self.txtrs_path.joinpath(self.txtrs_rfpaths[item])
        else:
            fpath_txtr = None
        frame = Pascal3DFrame(fpath_annotation=fpath_annotation, fpath_rgb=fpath_rgb, path_meshes=self.path_meshes, fpath_txtr=fpath_txtr)
        return frame

    @staticmethod
    def collate_fn(bla):
        frames = Pascal3DFrames(bla)
        return frames
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

class Pascal3DFrames:
    def __init__(self, frames: list[Pascal3DFrame]):
        frame0 = frames[0]
        self.dtype = frame0.dtype
        self.device = frame0.device
        self.rgb = torch.stack([frame.rgb for frame in frames], dim=0)
        self.mask = torch.stack([frame.mask for frame in frames], dim=0)
        self.depth = torch.stack([frame.depth for frame in frames], dim=0)
        self.cam_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0)
        self.cam_proj4x4_obj = torch.stack([frame.cam_proj4x4_obj for frame in frames], dim=0)
        self.cam_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0)
        self.kpts3d = torch.stack([frame.kpts3d for frame in frames], dim=0)
        self.kpts2d = torch.stack([frame.kpts2d for frame in frames], dim=0)
        self.kpts3d_vsbl = torch.stack([frame.kpts3d_vsbl for frame in frames], dim=0)
        self.kpts2d_annot = torch.stack([frame.kpts2d_annot for frame in frames], dim=0)
        self.kpts2d_annot_vsbl = torch.stack([frame.kpts2d_annot_vsbl for frame in frames], dim=0)

    def visualize(self):
        pts3d = torch.zeros(size=(1, 3)).to(device=self.device, dtype=self.dtype)
        mix_real_with_synthetic = blend_rgb(self.rgb[0], self.mask[0])

        mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic,
                                              proj3d2d(pts3d=torch.cat((pts3d, self.kpts3d[0, self.kpts3d_vsbl[0]])),
                                                       proj4x4=self.cam_proj4x4_obj[0]), colors=(0, 255, 0))
        mix_real_with_synthetic = draw_pixels(mix_real_with_synthetic, self.kpts2d_annot[0, self.kpts2d_annot_vsbl[0]],
                                              colors=(0, 0, 255), radius_in=2, radius_out=4)

        show_img(mix_real_with_synthetic)