import logging
logger = logging.getLogger(__name__)
import typer
import od3d.io
from pathlib import Path
import pandas as pd

app = typer.Typer()
from od3d.cli.benchmark import get_dataframe
from tabulate import tabulate
import re

@app.command()
def ablation_dist():
    # CO3Dv1_NeMo, metrics

    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    categories = od3d.io.read_config_intern(Path('datasets/categories/zsp.yaml'))

    align3d_1on1_name_partial = 'NeMo_Align3D_'
    align3d_1on1_metrics = ['pose/acc_pi6'] #, 'pose/acc_pi18']
    align3d_1on1_columns_map = {}
    align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = "Acc. Pi/6. [%]"
    #align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = "Acc. Pi/18. [%]"

    age_in_hours = 200
    configs = ['ablation_name']
    #configs = ['method.dist_appear_weight']
    #align3d_1on1_columns_map['method.dist_appear_weight'] = 'Appear. Weight'
    align3d_1on1_columns_map['ablation_name'] = 'Name'
    for category in categories:
        align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        # align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi18')
        align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = category
        #align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = category

    align3d_1on1_df = get_dataframe(configs=configs, metrics=align3d_1on1_metrics, age_in_hours=age_in_hours, name_partial=align3d_1on1_name_partial)
    #align3d_1on1_df['ablation_name'] = align3d_1on1_df['ablation_name'].str
    align3d_1on1_df = align3d_1on1_df.rename(columns=align3d_1on1_columns_map)

    cols_dec = ["Acc. Pi/6. [%]", 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster'] # , "Acc. Pi/18. [%]"
    align3d_1on1_df[cols_dec] *= 100.
    align3d_1on1_df = align3d_1on1_df.loc[align3d_1on1_df['Name'].notna()]
    align3d_1on1_df['Name'] = [row['value'] for row in align3d_1on1_df['Name']]
    cols = ['ablation_name', "Acc. Pi/6. [%]", 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster'] # , "Acc. Pi/18. [%]"

    cols = ["Acc. Pi/6. [%]", 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster'] #  "Acc. Pi/18. [%]",
    #cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'motorcycle', 'car', 'chair']

    align3d_1on1_df = align3d_1on1_df.set_index("Name")
    align3d_1on1_df = align3d_1on1_df.loc[[
        'resnet50_acc',
        'dino_vits8_acc',
        'dinov2_vitb14_acc',
        'dinov2_vits14_acc',
        # 'dinov2_dist_min',
        # 'dinov2_dist_avg',
        # 'dinov2_avg',
        # 'dinov2_avg_norm',
        # 'dinov2_dist_appear_weight_00',
        # 'dinov2_dist_appear_weight_01',
        # 'dinov2_dist_appear_weight_02',
        # 'dinov2_dist_appear_weight_03',
        # 'dinov2_dist_appear_weight_04',
        # 'dinov2_dist_appear_weight_05',
        # 'dinov2_dist_appear_weight_06',
        # 'dinov2_dist_appear_weight_07',
        # 'dinov2_dist_appear_weight_08',
        # 'dinov2_dist_appear_weight_09',
        # 'dinov2_dist_appear_weight_10',
    ]]

    df = align3d_1on1_df[cols]
    logger.info(tabulate(df, headers='keys', tablefmt='latex',  floatfmt=".1f")) # 'github', 'tsv', 'latex', 'latex_raw'

from typing import List
def get_categorical_results_from_multiple_runs(metrics, age_in_hours: float, configs=[], name_partial='_CO3D_NeMo_ref', metrics_scales=None):
    rows = []
    COLUMN_CATEGORY = "category"
    COLUMN_REFERENCE = "ref"
    COLUMN_CATEGORY_MEAN = "mean"
    COLUMN_INDEX = "index"
    df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours,
                                     name_regex=f'.*{name_partial}([0-9]*)_cat1_([a-z]*)_slurm', name_regex_groups=[COLUMN_REFERENCE, COLUMN_CATEGORY], filter_runs_with_metrics=False)
    metrics_dfs = []
    for m, metric in enumerate(metrics):
        if metrics_scales is not None and len(metrics_scales) > m:
            metric_scale = metrics_scales[m]
        else:
            metric_scale = 1.
        metric_df = df[df[metric].notnull()]
        metric_df = metric_df[[metric, COLUMN_CATEGORY, COLUMN_REFERENCE]]
        metric_df = metric_df.drop_duplicates(subset=[COLUMN_CATEGORY, COLUMN_REFERENCE], keep="first")
        metric_df_mean_over_refs = metric_df.groupby(COLUMN_CATEGORY)[metric].mean(numeric_only=False) * metric_scale
        metric_df_mean_over_refs[COLUMN_CATEGORY_MEAN] = metric_df_mean_over_refs.mean()
        metric_df_std_over_refs = metric_df.groupby(COLUMN_CATEGORY)[metric].std(numeric_only=False) * metric_scale
        metric_df_std_over_refs[COLUMN_CATEGORY_MEAN] = metric_df_std_over_refs.mean()
        metric_df = pd.DataFrame({'mean': metric_df_mean_over_refs, 'std': metric_df_std_over_refs}).reset_index()
        metric_df[COLUMN_INDEX] = 0
        metric_df = metric_df.pivot(index=COLUMN_INDEX, columns=COLUMN_CATEGORY, values=['mean', 'std'])
        metrics_dfs.append(metric_df)

    return metrics_dfs

# def get_categorical_and

@app.command()
def pose_pi6_categories_separate():
    COLUMN_ACC_PI6 = "Acc. Pi/6. [%]"
    logging.basicConfig(level=logging.INFO)
    #import wandb
    categories = od3d.io.read_config_intern(Path('datasets/categories/cross.yaml'))

    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_OD3D_TO_PASCAL3D
    #categories_co3d = [MAP_CATEGORIES_OD3D_TO_CO3D[cat] for cat in categories]
    #categories_pascal3d = [MAP_CATEGORIES_OD3D_TO_PASCAL3D[cat] for cat in categories]
    #config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm

    age_in_hours = 48
    configs = []
    metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/objectnet3d_test/pose/acc_pi6', 'test/co3d_5s_no_zsp_labeled/pose/acc_pi6', 'test/co3dv1_10s_zsp_labeled/pose/acc_pi6']
    metrics_scales = [100, 100, 100, 100]
    metrics_names = ['PASCAL3D [%]', 'ObjectNet3D [%]', 'CO3D 5s [%]', 'CO3D ZSP 10s [%]']

    metrics_dfs = get_categorical_results_from_multiple_runs(metrics=metrics, metrics_scales=metrics_scales, age_in_hours=age_in_hours, configs=configs, name_partial='_CO3D_NeMo_ref')
    for metric_df in metrics_dfs:
        #logger.info(metric_df.to_latex(float_format=".2f"))
        logger.info(tabulate(metric_df, headers='keys', tablefmt='latex', floatfmt=".2f"))

"""
@app.command()
def pose_pi6_categories_separate():
    COLUMN_ACC_PI6 = "Acc. Pi/6. [%]"
    logging.basicConfig(level=logging.INFO)
    #import wandb
    categories = od3d.io.read_config_intern(Path('datasets/categories/cross.yaml'))

    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_OD3D_TO_PASCAL3D
    #categories_co3d = [MAP_CATEGORIES_OD3D_TO_CO3D[cat] for cat in categories]
    #categories_pascal3d = [MAP_CATEGORIES_OD3D_TO_PASCAL3D[cat] for cat in categories]
    #config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm

    age_in_hours = 48
    configs = []
    metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/objectnet3d_test/pose/acc_pi6', 'test/co3d_5s_no_zsp_labeled/pose/acc_pi6', 'test/co3dv1_10s_zsp_labeled/pose/acc_pi6']
    metrics_names = ['PASCAL3D [%]', 'ObjectNet3D [%]', 'CO3D 5s [%]', 'CO3D ZSP 10s [%]']

    run_name_partial = '_CO3D_NeMo_cat1_'
    rows = []
    for i, metric in enumerate(metrics):
        metric_categories = get_categorical_results_from_multiple_runs(run_name_partial=run_name_partial, metric=metric, metric_name=metrics_names[i], age_in_hours=48, configs=[])
        logger.info(tabulate(metric_categories, headers='keys', tablefmt='latex', floatfmt=".3f"))
        rows.append(metric_categories)

    results = pd.concat(rows)
    print(results)

    # CO3Dv1_NeMo, metrics

    resutls = pd.concat(rows)
    nemo_name_partial = 'CO3Dv1_NeMo_local'
    nemo_metrics = ['val/co3d_10s_zsp_test/pose/acc_pi6']
    nemo_columns_map = {}
    nemo_columns_map[nemo_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        nemo_metrics.append(f'val/co3d_10s_zsp_test/pose/prefix/{category}_acc_pi6')
        nemo_columns_map[nemo_metrics[-1]] = category

    # CO3Dv1_NeMo_Align3D, metrics
    align3d_name_partial = 'CO3Dv1_NeMo_Align3D'
    align3d_metrics = ['only_to_ref/pose/acc_pi6']
    align3d_columns_map = {}
    align3d_columns_map[align3d_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        align3d_metrics.append(f'only_to_ref/pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        align3d_columns_map[align3d_metrics[-1]] = category

    # CO3Dv1_NeMo_Align3D, metrics
    align3d_1on1_name_partial = 'CO3Dv1_NeMo_Align3D'
    align3d_1on1_metrics = ['pose/acc_pi6']
    align3d_1on1_columns_map = {}
    align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = category


    nemo_df = get_dataframe(configs=configs, metrics=nemo_metrics, age_in_hours=age_in_hours, name_partial=nemo_name_partial)
    nemo_df = nemo_df.rename(columns=nemo_columns_map)
    align3d_df = get_dataframe(configs=configs, metrics=align3d_metrics, age_in_hours=age_in_hours, name_partial=align3d_name_partial)
    align3d_df = align3d_df.rename(columns=align3d_columns_map)
    align3d_1on1_df = get_dataframe(configs=configs, metrics=align3d_1on1_metrics, age_in_hours=age_in_hours, name_partial=align3d_1on1_name_partial)
    align3d_1on1_df = align3d_1on1_df.rename(columns=align3d_1on1_columns_map)

    df = pd.concat([nemo_df, align3d_df, align3d_1on1_df, pascal3d_nemo_df])
    cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster']
    cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'motorcycle', 'car', 'chair']

    df = df[cols]
    logger.info(tabulate(df, headers='keys', tablefmt='latex',  floatfmt=".3f")) # 'github', 'tsv', 'latex', 'latex_raw'
    # logger.info('\n' + my_df.to_csv(sep='\t', index=False, float_format="%.3f"))
    #logger.info('\n' + my_df.to_csv(sep=',', index=False, float_format="%.3f"))
"""

@app.command()
def pose_pi6_categories():
    logging.basicConfig(level=logging.INFO)
    #import wandb
    categories = od3d.io.read_config_intern(Path('datasets/categories/zsp.yaml'))

    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_OD3D_TO_PASCAL3D
    #categories_co3d = [MAP_CATEGORIES_OD3D_TO_CO3D[cat] for cat in categories]
    #categories_pascal3d = [MAP_CATEGORIES_OD3D_TO_PASCAL3D[cat] for cat in categories]
    #config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm

    age_in_hours = 24
    configs = []

    # CO3Dv1_NeMo, metrics
    pascal3d_nemo_name_partial = 'CO3Dv1_NeMo_local'
    pascal3d_nemo_metrics = ['test/pascal3d_test/pose/acc_pi6']
    pascal3d_nemo_columns_map = {}
    pascal3d_nemo_columns_map[pascal3d_nemo_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        pascal3d_nemo_metrics.append(f'test/pascal3d_test/pose/prefix/{category}_acc_pi6')
        pascal3d_nemo_columns_map[pascal3d_nemo_metrics[-1]] = category

    # CO3Dv1_NeMo, metrics
    nemo_name_partial = 'CO3Dv1_NeMo_local'
    nemo_metrics = ['val/co3d_10s_zsp_test/pose/acc_pi6']
    nemo_columns_map = {}
    nemo_columns_map[nemo_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        nemo_metrics.append(f'val/co3d_10s_zsp_test/pose/prefix/{category}_acc_pi6')
        nemo_columns_map[nemo_metrics[-1]] = category

    # CO3Dv1_NeMo_Align3D, metrics
    align3d_name_partial = 'CO3Dv1_NeMo_Align3D'
    align3d_metrics = ['only_to_ref/pose/acc_pi6']
    align3d_columns_map = {}
    align3d_columns_map[align3d_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        align3d_metrics.append(f'only_to_ref/pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        align3d_columns_map[align3d_metrics[-1]] = category

    # CO3Dv1_NeMo_Align3D, metrics
    align3d_1on1_name_partial = 'CO3Dv1_NeMo_Align3D'
    align3d_1on1_metrics = ['pose/acc_pi6']
    align3d_1on1_columns_map = {}
    align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = category

    pascal3d_nemo_df = get_dataframe(configs=configs, metrics=pascal3d_nemo_metrics, age_in_hours=age_in_hours, name_partial=pascal3d_nemo_name_partial)
    pascal3d_nemo_df = pascal3d_nemo_df.rename(columns=pascal3d_nemo_columns_map)
    nemo_df = get_dataframe(configs=configs, metrics=nemo_metrics, age_in_hours=age_in_hours, name_partial=nemo_name_partial)
    nemo_df = nemo_df.rename(columns=nemo_columns_map)
    align3d_df = get_dataframe(configs=configs, metrics=align3d_metrics, age_in_hours=age_in_hours, name_partial=align3d_name_partial)
    align3d_df = align3d_df.rename(columns=align3d_columns_map)
    align3d_1on1_df = get_dataframe(configs=configs, metrics=align3d_1on1_metrics, age_in_hours=age_in_hours, name_partial=align3d_1on1_name_partial)
    align3d_1on1_df = align3d_1on1_df.rename(columns=align3d_1on1_columns_map)

    df = pd.concat([nemo_df, align3d_df, align3d_1on1_df, pascal3d_nemo_df])
    cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster']
    cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'motorcycle', 'car', 'chair']

    df = df[cols]
    logger.info(tabulate(df, headers='keys', tablefmt='latex',  floatfmt=".3f")) # 'github', 'tsv', 'latex', 'latex_raw'
    # logger.info('\n' + my_df.to_csv(sep='\t', index=False, float_format="%.3f"))
    #logger.info('\n' + my_df.to_csv(sep=',', index=False, float_format="%.3f"))
