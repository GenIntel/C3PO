import torch
from od3d.cv.geometry.transform import reproj2d3d_broadcast, so3_exp_map, rot3x3
from logging import getLogger
logger = getLogger(__file__)


def axis3d_from_pxl2d(kpts2d_orient, cam_intr4x4):
    orients = {}

    A = []  # torch.zeros(size=(6, 9))
    for key in kpts2d_orient.keys():
        kpts2d_orient_lr = kpts2d_orient[key].clone()
        if len(kpts2d_orient_lr) > 0:
            kpts3d_homog_orient_lr = reproj2d3d_broadcast(kpts2d_orient_lr, cam_intr4x4.inverse())

            vector_homog_orient_lr = (kpts3d_homog_orient_lr[:, 0] - kpts3d_homog_orient_lr[:, 1])
            vector_homog_orient_lr = vector_homog_orient_lr / vector_homog_orient_lr.norm(dim=-1, keepdim=True)

            kpts3d_homog_orient_lr = kpts3d_homog_orient_lr / kpts3d_homog_orient_lr.norm(dim=-1, keepdim=True)

            orient3d_plane1 = torch.linalg.cross(kpts3d_homog_orient_lr[0, 0], kpts3d_homog_orient_lr[0, 1])
            orient3d_plane1 = orient3d_plane1 / orient3d_plane1.norm(dim=-1, keepdim=True)
            orient3d_plane2 = torch.linalg.cross(kpts3d_homog_orient_lr[1, 0], kpts3d_homog_orient_lr[1, 1])
            orient3d_plane2 = orient3d_plane2 / orient3d_plane2.norm(dim=-1, keepdim=True)
            orient3d = torch.linalg.cross(orient3d_plane1, orient3d_plane2)
            orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
            eins = torch.einsum('d,cd->c', orient3d, vector_homog_orient_lr[:])  # [:1])

            A_kp = torch.zeros(size=(2, 9))
            if key == 'left-right':
                A_kp[0, :3] = orient3d_plane1
                A_kp[1, :3] = orient3d_plane2
            elif key == 'back-front':
                A_kp[0, 3:6] = orient3d_plane1
                A_kp[1, 3:6] = orient3d_plane2
            elif key == 'top-bottom':
                A_kp[0, 6:9] = orient3d_plane1
                A_kp[1, 6:9] = orient3d_plane2
            A.append(A_kp)
            if (eins < 0.).all():
                logger.info(f'switch axis for {key}')
                orients[key] = -orient3d
            elif (eins > 0.).all():
                orients[key] = orient3d
            else:
                logger.warning(f'Inconsistency of axis3d with axis2d alignment. Skipping orient for {key}...')
                orients[key] = []
        else:
            orients[key] = []
    if len(orients['left-right']) == 0:
        orient3d = torch.linalg.cross(orients['back-front'], orients['top-bottom'])
        orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        orients['left-right'] = orient3d
    if len(orients['back-front']) == 0:
        orient3d = torch.linalg.cross(orients['top-bottom'], orients['left-right'])
        orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        orients['back-front'] = orient3d
    if len(orients['top-bottom']) == 0:
        orient3d = torch.linalg.cross(orients['left-right'], orients['back-front'])
        orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        orients['top-bottom'] = orient3d

    orients = torch.stack([orients['left-right'], orients['back-front'], orients['top-bottom']], dim=0)
    # cam_rot3x3_obj = so3_exp_map(so3_log_map(orients.T))

    cam_rot3x3_obj = orients.T
    A = torch.cat(A, dim=0)
    tmp_rot3_obj = torch.zeros(3).to(device=cam_rot3x3_obj.device)

    for i in range(100):
        cam_rot3x3_obj = rot3x3(cam_rot3x3_obj.detach(), so3_exp_map(tmp_rot3_obj.detach()))

        tmp_rot3_obj = torch.nn.Parameter(torch.zeros(3).to(device=cam_rot3x3_obj.device), requires_grad=True)
        optimizer = torch.optim.SGD(params=[tmp_rot3_obj], lr=0.01)

        cam_rot3x3_obj = rot3x3(cam_rot3x3_obj, so3_exp_map(tmp_rot3_obj))
        loss = (torch.bmm(A[None,], cam_rot3x3_obj.T.reshape(1, 9, 1))[0]).norm(dim=-1).mean(dim=-1) + (
                    rot3x3(cam_rot3x3_obj.T, cam_rot3x3_obj) - torch.eye(3)).norm(dim=-1).mean(dim=-1)
        loss.backward()
        # logger.info(f"Loss {loss}, cam_rot3x3_obj {cam_rot3x3_obj}")
        optimizer.step()


    return cam_rot3x3_obj.detach()

