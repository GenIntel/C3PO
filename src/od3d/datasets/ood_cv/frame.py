import logging

import torchvision.io.image

logger = logging.getLogger(__name__)
import torch
from typing import List
from pathlib import Path
from dataclasses import dataclass
from od3d.datasets.pascal3d.enum import (
    PASCAL3D_SCALE_NORMALIZE_TO_REAL,
)
from od3d.datasets.pascal3d.frame import Pascal3DFrame, Pascal3DFrameMeta
from od3d.cv.geometry.objects3d.meshes import Mesh
import torchvision
from od3d.datasets.pascal3d.enum import (
    MAP_CATEGORIES_OD3D_TO_PASCAL3D,
)
from od3d.datasets.frame import (
    OD3D_FRAME_MASK_TYPES,
    OD3D_FRAME_DEPTH_TYPES,
    OD3D_FRAME_KPTS2D_ANNOT_TYPES,
)
from od3d.datasets.object import (
    OD3D_CAM_TFORM_OBJ_TYPES,
    OD3D_FRAME_MASK_TYPES,
    OD3D_MESH_TYPES,
    OD3D_MESH_FEATS_TYPES,
    OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
    OD3D_TFROM_OBJ_TYPES,
)


@dataclass
class OOD_CV_FrameMeta(Pascal3DFrameMeta):
    @staticmethod
    def load_from_raw_annotation(
        annotation,
        subset: str,
        category: str,
        path_raw: Path,
        rpath_meshes: Path,
        rfpath_rgb: Path,
        path_pascal3d_raw: Path = None,
    ):
        name = annotation["record"]["filename"][0][0][0].split(".")[0]

        objects = annotation["record"]["objects"][0][0][0]

        objects = list(
            filter(
                lambda obj: hasattr(obj, "dtype")
                and obj.dtype.names is not None
                and "viewpoint" in obj.dtype.names
                and hasattr(obj["viewpoint"], "dtype")
                and obj["viewpoint"].dtype.names is not None,
                objects,
            ),
        )
        # assert len(objects) == 1
        if len(objects) < 1:
            incomplete_reason = f"num objects = {len(objects)}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        object = objects[0]

        if (
            not hasattr(object, "dtype")
            or object.dtype.names is None
            or "viewpoint" not in object.dtype.names
        ):  # ['focal'][0][0][0][0] == 0:
            incomplete_reason = "viewpoint missing"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        if (
            not hasattr(object["viewpoint"], "dtype")
            or object["viewpoint"].dtype.names is None
        ):  # ['focal'][0][0][0][0] == 0:
            incomplete_reason = "viewpoint missing"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        # if 'focal' not in object['viewpoint'].dtype.names or object['viewpoint']['focal'][0][0][0][0] == 0:
        #    object['viewpoint']['focal'][0][0][0][0] = 3000

        if object["viewpoint"]["px"][0][0][0][0] < 0:
            incomplete_reason = f"negative px {object['viewpoint']['px'][0][0][0][0]}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None
            # object['viewpoint']['px'][0][0][0][0] = -object['viewpoint']['px'][0][0][0][0]

        if object["viewpoint"]["py"][0][0][0][0] < 0:
            incomplete_reason = f"negative py {object['viewpoint']['py'][0][0][0][0]}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None
            # object['viewpoint']['py'][0][0][0][0] = -object['viewpoint']['py'][0][0][0][0]

        if object["viewpoint"]["distance"][0][0][0][0] < 0.01:
            incomplete_reason = (
                f"distance negative {object['viewpoint']['distance'][0][0][0][0]}"
            )
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        if (
            not hasattr(object["anchors"][0][0], "dtype")
            or object["anchors"][0][0].dtype.names is None
        ):
            incomplete_reason = "kpts names missing"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        fpath_rgb = Path(path_raw).joinpath(rfpath_rgb)
        img = torchvision.io.image.read_image(str(fpath_rgb))
        size = torch.LongTensor([*img.shape[1:]])

        # category changing the name if abstracted from .mat file of ood cv ( e.g. 'diningtable' -> 'table')
        (
            category,
            mesh_index,
            rfpath_mesh,
            bbox,
            kpts_names,
            kpts2d_annot,
            kpts2d_annot_vsbl,
            cam_tform4x4_obj,
            cam_intr4x4,
        ) = Pascal3DFrameMeta.load_category_mesh_bbox_kpts2d_cam_from_object_annotation_raw(
            object=object,
            rpath_meshes=rpath_meshes,
            category=category,
        )

        # logger.info(cam_intr4x4)

        kpts3d = Pascal3DFrameMeta.load_kpts3d_from_raw(
            path_pascal3d_raw,
            rpath_meshes,
            category,
            mesh_index,
            kpts_names,
        )

        return OOD_CV_FrameMeta(
            subset=subset,
            name=name,
            rfpath_rgb=rfpath_rgb,
            rfpath_mesh=rfpath_mesh,
            l_bbox=bbox.tolist(),
            kpts_names=kpts_names,
            l_kpts2d_annot=kpts2d_annot.tolist(),
            l_kpts2d_annot_vsbl=kpts2d_annot_vsbl.tolist(),
            l_size=size.tolist(),
            l_cam_tform4x4_obj=cam_tform4x4_obj.tolist(),
            l_cam_intr4x4=cam_intr4x4.tolist(),
            l_kpts3d=kpts3d.tolist(),
            category=category,
        )


class OOD_CV_Frame(Pascal3DFrame):
    #'name_unique', 'all_categories', 'depth_type', 'mask_type', 'cam_tform4x4_obj_type', 'kpts2d_annot_type', 'tform_obj_type', 'mesh_type', 'mesh_feats_type', and 'mesh_feats_dist_reduce_type' '''
    MAP_OD3D_CATEGORIES = MAP_CATEGORIES_OD3D_TO_PASCAL3D

    def __init__(
        self,
        path_raw: Path,
        path_preprocess: Path,
        modalities,
        name_unique: str,
        path_meshes: Path,
        all_categories: List[str],
        depth_type: OD3D_FRAME_DEPTH_TYPES,
        mask_type: OD3D_FRAME_MASK_TYPES,
        cam_tform4x4_obj_type: OD3D_CAM_TFORM_OBJ_TYPES,
        kpts2d_annot_type: OD3D_FRAME_KPTS2D_ANNOT_TYPES,
        tform_obj_type: OD3D_TFROM_OBJ_TYPES,
        mesh_type: OD3D_MESH_TYPES,
        mesh_feats_type: OD3D_MESH_FEATS_TYPES,
        mesh_feats_dist_reduce_type: OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
    ):
        super().__init__(
            path_raw=path_raw,
            path_preprocess=path_preprocess,
            modalities=modalities,
            name_unique=name_unique,
            all_categories=all_categories,
            depth_type=depth_type,
            mask_type=mask_type,
            cam_tform4x4_obj_type=cam_tform4x4_obj_type,
            kpts2d_annot_type=kpts2d_annot_type,
            tform_obj_type=tform_obj_type,
            mesh_type=mesh_type,
            mesh_feats_type=mesh_feats_type,
            mesh_feats_dist_reduce_type=mesh_feats_dist_reduce_type,
        )
        self.path_meshes = path_meshes

    @staticmethod
    def get_rfpath_pp_categorical_mesh(mesh_type: OD3D_MESH_TYPES, category: str):
        return Path("mesh", f"{mesh_type}", f"{category}", "mesh.ply")

    def get_fpath_mesh(self, mesh_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_type == OD3D_MESH_TYPES.META:
            return self.path_meshes.joinpath(self.meta.rfpath_mesh)
        else:
            return self.path_preprocess.joinpath(
                self.get_rfpath_pp_categorical_mesh(
                    mesh_type=mesh_type,
                    category=self.category,
                ),
            )

    def read_mesh(self, mesh_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_type == OD3D_MESH_TYPES.META:
            mesh = Mesh.load_from_file(
                fpath=self.get_fpath_mesh(mesh_type=mesh_type),
                scale=PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category],
            )
        else:
            # note: preprocessed meshes are in real scale
            mesh = Mesh.load_from_file(fpath=self.get_fpath_mesh(mesh_type=mesh_type))

        if mesh_type is None or mesh_type == self.mesh_type:
            self.mesh = mesh
        return mesh

    def get_mesh(self, mesh_type=None, clone=False):
        if (mesh_type is None or mesh_type == self.mesh_type) and self.mesh is not None:
            mesh = self.mesh
        else:
            mesh = self.read_mesh(mesh_type=mesh_type)

        if not clone:
            return mesh
        else:
            return mesh.clone()
