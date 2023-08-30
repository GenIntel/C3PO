import logging

from sympy import yn

logger = logging.getLogger(__name__)
import torch
from typing import Dict, List, Union
import wandb
import math

from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay
import numpy as np

class OD3D_Results(Dict[str, Union[torch.Tensor, List]]):
    def __init__(self, device: torch.device='cpu', init_dict: Dict[str, Union[torch.Tensor, List]]=None):
        super().__init__()
        self.mean_blocklist = ['label_gt', 'label_pred', 'rot_diff_rad', 'name_unique', 'item_id', 'cam_tform4x4_obj', 'label_names']
        self.log_blocklist = ['name_unique', 'item_id', 'cam_tform4x4_obj']
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
            if 'label_names' in self.keys():
                label_names = self['label_names']
            else:
                label_names = [str(i) for i in range(max(set(self['label_gt'] + self['label_pred']))+1)]
            res['label/acc'] = (self['label_gt'] == self['label_pred']).to(dtype=float).mean(dim=0)
            res['label/confusion'] = wandb.plot.confusion_matrix(probs=None,
                                                                 y_true=self['label_gt'].numpy(), preds=self['label_pred'].numpy(),
                                                                 class_names=label_names)

        if 'rot_diff_rad' in self.keys():
            if 'sim' in self.keys():
                res['pose/roc/pi6'] = self.get_roc(ground_truth=(self['rot_diff_rad'] < math.pi / 6.).numpy().astype(np.int), predictions=self['sim'][:, 0].detach().numpy(), title="PI/6 ROC: TPR vs FPR")
                res['pose/roc/pi18'] = self.get_roc(ground_truth=(self['rot_diff_rad'] < math.pi / 18.).numpy().astype(np.int), predictions=self['sim'][:, 0].detach().numpy(), title="PI/18 ROC: TPR vs FPR")

            res['pose/acc_pi6'] = (self['rot_diff_rad'] < math.pi / 6.).to(dtype=float).mean()
            res['pose/acc_pi18'] = (self['rot_diff_rad'] < math.pi / 18.).to(dtype=float).mean()
            res['pose/err_median'] = 180 / math.pi * self['rot_diff_rad'].median()
            res['pose/err_mean'] = 180 / math.pi * self['rot_diff_rad'].mean()

        return OD3D_Results(init_dict=res)

    def get_roc(self, ground_truth, predictions, title): #labels, predictions, positive_label, thresholds_every=10, title=''):

        # fp: false positive rates. tp: true positive rates
        thresholds_every=1
        fp, tp, thresholds = metrics.roc_curve(ground_truth, predictions)
        roc_auc = metrics.auc(fp, tp)

        plt.ioff()
        # Create a Figure object
        fig, ax = plt.subplots(figsize=(4, 4))
        # fig = plt.figure(figsize=(16, 16))

        # Add a subplot (1 row, 1 column, first subplot)
        #ax = fig.add_subplot(1, 1, 1)

        ax.axis("square")
        ax.plot(fp, tp, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=2, color='darkorange')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2)
        ax.set_xlabel('False positives rate')
        ax.set_ylabel('True positives rate')
        ax.set_xlim([-0.03, 1.0])
        ax.set_ylim([0.0, 1.03])
        ax.set_title(title)
        ax.legend()
        # ax.legend(loc="lower right")
        # ax.grid(True)

        # plot some thresholds
        thresholdsLength = len(thresholds)
        colorMap = plt.get_cmap('jet', thresholdsLength)
        for i in range(0, thresholdsLength, thresholds_every):
            if np.isfinite(fp[i]) and np.isfinite(tp[i]):
                threshold_value_with_max_four_decimals = str(thresholds[i])[:5]
                ax.text(fp[i] - 0.03, tp[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 10},
                         color=colorMap(i / thresholdsLength))

        from od3d.cv.visual.show import get_img_from_plot
        from od3d.cv.io import image_as_wandb_image
        img = get_img_from_plot(ax=ax, fig=fig, axis_off=False)
        plt.close(fig)
        return image_as_wandb_image(img, caption=title)

    #
    # def get_roc(self, ground_truth, predictions, title):
    #     """
    #         Args:
    #             ground_truth (np.ndarray): (n_samples,), int {0, 1}
    #             predictions (np.ndarray): (n_samples,), float
    #         Returns:
    #             figure (matplotlib figure)
    #     """
    #     display = RocCurveDisplay.from_predictions(
    #         ground_truth,
    #         predictions,
    #         name=title,
    #         drop_intermediate=False,
    #     )
    #     display.ax_.axis("square")
    #     display.ax_.set_xlabel("False Positive Rate")
    #     display.ax_.set_ylabel("True Positive Rate")
    #     display.ax_.set_title(title)
    #     display.ax_.legend()
    #     return display.figure_
    #
    #
    #     # table = wandb.Table(data=data, columns=["false positive rate", "true positive rate"])
    #     # return wandb.plot.line_series(
    #     #     xs=[0, 1, 2, 3, 4],
    #     #     ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
    #     #     keys=["metric Y", "metric Z"],
    #     #     title=title,
    #     #     xname="false positive rate",
    #     #     yn="true positive rate")
    #
    #     # return wandb.plot.line(table, "false positive rate", "true positive rate", title=title)


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