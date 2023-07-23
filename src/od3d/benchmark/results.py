import logging
logger = logging.getLogger(__name__)
import torch
from typing import Dict, List, Union
import wandb
import math


class OD3D_Results(Dict[str, Union[torch.Tensor, List]]):
    def __init__(self, device: torch.device='cpu', init_dict: Dict[str, Union[torch.Tensor, List]]=None):
        super().__init__()
        self.mean_blocklist = ['label_gt', 'label_pred', 'rot_diff_rad', 'name_unique', 'item_id']
        self.log_blocklist = ['name_unique', 'item_id']
        self.device = device

        if init_dict is not None:
           self.__add__(other=init_dict)

    def __add__(self, other: Dict[str, torch.Tensor]):
        for key, val in other.items():
            if isinstance(val, torch.Tensor):
                if val.dim() == 0:
                    val = val[None,]
                if key not in self.keys():
                    self[key] = val.to(device=self.device) # torch.Tensor(size=val.shape, device=self.device, dtype=val.dtype)
                else:
                    self[key] = torch.cat([self[key], val.to(device=self.device)], dim=0)
            elif isinstance(val, List):
                if key not in self.keys():
                    self[key] = []
                self[key] += val
            else:
                self[key] = val
        return self

    def mean(self):
        res = {}
        for key, val in self.items():
            if key not in self.mean_blocklist:
                res[key] = val.mean(dim=0)

        if 'label_gt' in self.keys() and 'label_pred' in self.keys():
            res['label/acc'] = (self['label_gt'] == self['label_pred']).to(dtype=float).mean(dim=0)

        if 'rot_diff_rad' in self.keys():
            res['pose/acc_pi6'] = (self['rot_diff_rad'] < math.pi / 6.).to(dtype=float).mean()
            res['pose/acc_pi18'] = (self['rot_diff_rad'] < math.pi / 18.).to(dtype=float).mean()
            res['pose/err_median'] = 180 / math.pi * self['rot_diff_rad'].median()
            res['pose/err_mean'] = 180 / math.pi * self['rot_diff_rad'].mean()

        return OD3D_Results(init_dict=res)

    def get_filtered_log_results(self):
        res = {}
        for key, val in self.items():
            if key not in self.log_blocklist:
                res[key] = val
        return res

    def log(self):
        wandb.log(self.get_filtered_log_results())

    def log_with_prefix(self, prefix: str):
        wandb.log({prefix + '/' + k: v for k, v in self.get_filtered_log_results().items()})