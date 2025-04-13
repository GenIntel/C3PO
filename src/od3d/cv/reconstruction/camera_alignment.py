import torch
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def skew(vector: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [torch.zeros_like(vector[..., 0]), -vector[..., 2], vector[..., 1]],
                dim=-1,
            ),
            torch.stack(
                [vector[..., 2], torch.zeros_like(vector[..., 0]), -vector[..., 0]],
                dim=-1,
            ),
            torch.stack(
                [-vector[..., 1], vector[..., 0], torch.zeros_like(vector[..., 0])],
                dim=-1,
            ),
        ],
        dim=-1,
    )


def logarithmic_map(rotation: torch.Tensor) -> torch.Tensor:
    # rotation torch.Tensor of shape (B, 3, 3)
    angle = torch.arccos(
        (torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1) / 2.0
    ).unsqueeze(-1)
    axis = (
        1
        / (2 * torch.sin(angle))
        * torch.stack(
            [
                rotation[..., 2, 1] - rotation[..., 1, 2],
                rotation[..., 0, 2] - rotation[..., 2, 0],
                rotation[..., 1, 0] - rotation[..., 0, 1],
            ],
            dim=-1,
        )
    )
    return angle * axis
def safe_logarithmic_map(R: torch.Tensor, eps=1e-7) -> torch.Tensor:
    # R: [3, 3]
    def vee(S: torch.Tensor) -> torch.Tensor:
        """
        Converts a 3x3 skew-symmetric matrix (or batch of them) back into a 3D vector (or batch).
        For a skew-symmetric matrix:
            [  0   -z    y ]
            [  z    0   -x ]
            [ -y    x    0 ]
        vee(S) = (x, y, z).
        """
        x = S[..., 2, 1] - S[..., 1, 2]
        y = S[..., 0, 2] - S[..., 2, 0]
        z = S[..., 1, 0] - S[..., 0, 1]
        return torch.stack([x, y, z], dim=-1)
    trace = R.trace()
    # clamp trace-based cos_theta to [-1, 1]
    cos_theta = max(min((trace - 1.0) / 2.0, 1.0), -1.0)

    theta = torch.arccos(cos_theta)
    if theta < eps:
        # Use small-angle approximation: 
        # R ~ I + skew(omega), so (R - I) ~ skew(omega)
        # => omega ~ vee(R - I)/2
        return 0.5 * vee(R - torch.eye(3, device=R.device))
    else:
        # Standard formula
        axis = (1.0 / (2.0 * torch.sin(theta))) * torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], dim=0)
        return theta * axis



def exponential_map(so3: torch.Tensor) -> torch.Tensor:
    # so3 torch.Tensor of shape (B, 3)
    theta = torch.norm(so3, dim=-1, keepdim=True)
    axis = so3 / theta
    axis_skew = skew(axis)
    return (
        torch.eye(3, device=so3.device).unsqueeze(0).expand(so3.shape[0], -1, -1)
        + torch.sin(theta).unsqueeze(-1) * axis_skew
        + (1 - torch.cos(theta).unsqueeze(-1)) * axis_skew @ axis_skew
    )


def calculate_rotational_offset(co3d_transforms, colmap_transforms):
    # co3d_transforms torch.Tensor of shape (B, 4, 4)
    # colmap_transforms torch.Tensor of shape (B, 4, 4)
    return exponential_map(
        torch.mean(
            logarithmic_map(
                co3d_transforms[..., :3, :3]
                @ colmap_transforms[:, :3, :3].transpose(-2, -1)
            ),
            dim=0,
            keepdim=True,
        )
    )
    # return exponential_map(
    #     torch.mean(
    #         safe_logarithmic_map(
    #             co3d_transforms[..., :3, :3]
    #             @ colmap_transforms[:, :3, :3].transpose(-2, -1)
    #         ),
    #         dim=0,
    #         keepdim=True,
    #     )
    # )
    


def calulate_scale_offset(co3d_transforms, colmap_transforms):
    # co3d_transforms torch.Tensor of shape (B, 3, 3)
    # colmap_transforms torch.Tensor of shape (B, 3, 3)
    co3d_translations = co3d_transforms[:, :3, 3]
    colmap_translations = colmap_transforms[:, :3, 3]
    co3d_mean = torch.mean(co3d_translations, dim=0, keepdim=True)
    colmap_mean = torch.mean(colmap_translations, dim=0, keepdim=True)

    # version 1
    # s = (torch.mean(torch.norm(co3d_translations - co3d_mean, dim=-1))) / (
    #     (torch.mean(torch.norm(colmap_translations - colmap_mean, dim=-1)))
    # )

    # # version 2
    s = torch.mean(
        torch.norm(co3d_translations - co3d_mean)
        / torch.norm(colmap_translations - colmap_mean)
    )

    return s


def calculate_translational_offset(co3d_transforms, colmap_transforms):
    # co3d_transforms torch.Tensor of shape (B, 4, 4)
    # colmap_transforms torch.Tensor of shape (B, 4, 4)
    return torch.mean(
        co3d_transforms[:, :3, 3] - colmap_transforms[:, :3, 3], dim=0, keepdim=True
    )
def is_3x3_with_nans(tensor: torch.Tensor) -> bool:
    # Check shape is (1, 3, 3)
    if tensor.shape != (1, 3, 3):
        return False
    
    # Check if there are any NaNs in the tensor
    return torch.isnan(tensor).any().item()

def calculate_offset(co3d_transforms, colmap_transforms):
    rotational_offset = calculate_rotational_offset(co3d_transforms, colmap_transforms)
    if is_3x3_with_nans(rotational_offset):
        rotational_offset = torch.eye(3).unsqueeze(0).cuda()
    new_transforms = colmap_transforms.clone()
    new_transforms[:, :3, :3] = (
        rotational_offset.transpose(-2, -1) @ new_transforms[:, :3, :3]
    )
    rotated_co3d = co3d_transforms.clone()
    rotated_co3d[:, :3, :3] = rotational_offset @ rotated_co3d[:, :3, :3]
    rotated_co3d[:, :3, 3] = (rotational_offset @ rotated_co3d[:, :3, 3].unsqueeze(-1))[
        ..., 0
    ]
    scale_offset = calulate_scale_offset(co3d_transforms, new_transforms)
    new_transforms[:, :3, 3] = scale_offset * new_transforms[:, :3, 3]
    translational_offset = -calculate_translational_offset(rotated_co3d, new_transforms)
    new_transforms[:, :3, 3] = translational_offset + new_transforms[:, :3, 3]

    return new_transforms, rotational_offset, 1 / scale_offset, translational_offset
