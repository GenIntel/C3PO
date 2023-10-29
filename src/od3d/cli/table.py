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
                                     name_regex=f'.*{name_partial}([0-9]*)_cat1_([a-z]*).*', name_regex_groups=[COLUMN_REFERENCE, COLUMN_CATEGORY], filter_runs_with_metrics=False)
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
        metric_df = metric_df.set_index(COLUMN_CATEGORY).transpose()
        metrics_dfs.append(metric_df)

    return metrics_dfs

def get_categorical_results_from_single_runs(metrics, categories, age_in_hours: float, configs=[], name_partial='_CO3D_NeMo_ref', metrics_scales=None, map_od3d_to_datasets=None):
    metrics_df = []
    for m, metric in enumerate(metrics):
        columns_map = {}
        metrics = []  # ['pose/acc_pi6', 'pose/acc_pi6_std']
        columns_std = []
        columns_mean = []
        columns_mean_map = {}
        columns_std_map = {}

        if map_od3d_to_datasets is None or len(map_od3d_to_datasets) <= m:
            map_od3d_to_dataset = {}
        else:
            map_od3d_to_dataset = map_od3d_to_datasets[m]
        for category in categories:
            if category not in map_od3d_to_dataset:
                map_od3d_to_dataset[category] = category # f'pose/prefix/{map_od3d_to_dataset[category]}_acc_pi6' # f'pose/prefix/{map_od3d_to_dataset[category]}_acc_pi6_std'
            metrics.append(metric.replace('CATEGORY', map_od3d_to_dataset[category]))
            columns_map[metrics[-1]] = category + '_mean'
            columns_mean.append(columns_map[metrics[-1]])
            columns_mean_map[columns_map[metrics[-1]]] = category
            metrics.append(metric.replace('CATEGORY', map_od3d_to_dataset[category]) + '_std')
            columns_map[metrics[-1]] = category + '_std'
            columns_std.append(columns_map[metrics[-1]])
            columns_std_map[columns_map[metrics[-1]]] = category

        df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours, name_regex=f'.*{name_partial}.*')
        df = df.rename(columns=columns_map)

        if metrics_scales is not None and len(metrics_scales) > m:
            metric_scale = metrics_scales[m]
        else:
            metric_scale = 1.
        df = df * metric_scale
        df = df.iloc[-1:] # [:1] or [-1:]
        df = pd.concat([df[columns_mean].rename(columns=columns_mean_map), df[columns_std].rename(columns=columns_std_map)])
        df['category'] = ['mean', 'std']
        df = df.set_index('category')
        metrics_df.append(df)
        # metric_df = metric_df.set_index(COLUMN_CATEGORY).transpose()
    return metrics_df

# def get_categorical_and
TABLE_CATEGORIES_OBJECTNET3D_3 = ['cellphone', 'toilet', 'microwave', 'mean (3)']
TABLE_CATEGORIES_OBJECTNET3D_23 = [
    'cellphone', 'toilet', 'microwave', 'airplane', 'backpack', 'bench', 'bicycle', 'bottle', 'bus', 'car',
    'cellphone', 'chair', 'couch', 'cup', 'hairdryer', 'keyboard', 'laptop', 'microwave', 'motorcycle',
    'mouse', 'remote', 'suitcase', 'toaster', 'toilet', 'train', 'tv', 'mean (23)'
]
TABLE_CATEGORIES_CO3D_20 = ['bicycle', 'truck', 'train', 'teddybear', 'car', 'bus', 'motorcycle', 'keyboard', 'handbag', 'remote', 'airplane', 'toilet', 'hairdryer', 'mouse', 'toaster', 'hydrant', 'chair', 'laptop', 'book', 'backpack', 'mean (20)']
TABLE_CATEGORIES_CO3D_28 = ['bicycle', 'truck', 'train', 'teddybear', 'car', 'bus', 'motorcycle', 'keyboard', 'handbag', 'remote', 'airplane', 'toilet', 'hairdryer', 'mouse', 'toaster', 'hydrant', 'chair', 'laptop', 'book', 'backpack', 'cellphone', 'microwave', 'bench', 'bottle', 'couch', 'cup', 'suitcase', 'tv', 'mean (28)']

TABLE_CATEGORIES_ZSP = ['bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster', 'mean (20)']
TABLE_CATEGORIES_YOLO = ['backpack', 'car', 'chair', 'keyboard', 'laptop', 'motorcycle', 'mean (20)'] # # B’pack Car Chair Keyboard Laptop M’cycle
TABLE_CATEGORIES_5S = ['mean (20)', 'mean (28)']
TABLE_CATEGORIES_PASCAL3D = ['airplane', 'bicycle', 'bottle', 'bus', 'car', 'chair', 'motorcycle', 'couch', 'train', 'tv', 'mean (10)']

DATASET_PASCAL3D = 'pascal3d'
DATASET_CO3D_20 = 'co3d_20'
DATASET_CO3D_28 = 'co3d_28'
DATASET_OBJECTNET3D = 'objectnet3d'

@app.command()
def pose_pi6_categories_separate():
    COLUMN_ACC_PI6 = "Acc. Pi/6. [%]"
    logging.basicConfig(level=logging.INFO)
    #import wandb
    categories28 = od3d.io.read_config_intern(Path('datasets/categories/cross.yaml'))
    categories20 = od3d.io.read_config_intern(Path('datasets/categories/zsp.yaml'))
    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_OD3D_TO_PASCAL3D
    #categories_co3d = [MAP_CATEGORIES_OD3D_TO_CO3D[cat] for cat in categories]
    #categories_pascal3d = [MAP_CATEGORIES_OD3D_TO_PASCAL3D[cat] for cat in categories]
    #config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm
    # voge:    bed shelf     calculator cellphone computer cabinet        guitar iron knife oven      pen pot rifle slipper stove toilet tub wheelchair
    # starmap: bed bookshelf calculator cellphone computer filing cabinet guitar iron knife microwave pen pot rifle slipper stove toilet tub wheelchair
                # aero bike boat bottle bus car chair table mbike sofa train tv mean
    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D


    age_in_hours = 12
    configs = []

    ###### FROM MULTIPLE RUNS
    metrics_dataset = [DATASET_PASCAL3D, DATASET_OBJECTNET3D, DATASET_CO3D_28, DATASET_CO3D_20]
    metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/objectnet3d_test/pose/acc_pi6', 'test/co3d_5s_no_zsp_labeled/pose/acc_pi6', 'test/co3dv1_10s_zsp_labeled/pose/acc_pi6']
    metrics_scales = [100, 100, 100, 100]
    metrics_names = ['PASCAL3D [%]', 'ObjectNet3D [%]', 'CO3D 5s [%]', 'CO3D ZSP 10s [%]']
    name_partial = '_CO3D_NeMo_ref'
    #name_partial = '_CO3D_NeMo_Incremental_ref'
    metrics_dfs = get_categorical_results_from_multiple_runs(metrics=metrics, metrics_scales=metrics_scales, age_in_hours=age_in_hours, configs=configs, name_partial=name_partial)

    ###### FROM SINGLE RUNS
    # metrics = ['pose/prefix/CATEGORY_acc_pi6']
    # metrics_dataset = [DATASET_CO3D_20]
    # metrics_dataset = [DATASET_CO3D_28]
    # metrics_scales = [100]
    # categories = TABLE_CATEGORIES_CO3D_20[:-1]
    # categories = TABLE_CATEGORIES_CO3D_28[:-1]
    #
    # map_od3d_to_datasets = [MAP_CATEGORIES_OD3D_TO_CO3D]
    # map_od3d_to_datasets = [MAP_CATEGORIES_OD3D_TO_CO3D]
    #
    # name_partial = '_CO3Dv1_NeMo_Align3D_'
    # name_partial = '_CO3D_NeMo_Align3D_'
    # metrics_dfs = get_categorical_results_from_single_runs(metrics=metrics, categories=categories, age_in_hours=age_in_hours, name_partial=name_partial, metrics_scales=metrics_scales, map_od3d_to_datasets=map_od3d_to_datasets)


    for m, metric_df in enumerate(metrics_dfs):
        if metrics_dataset[m] == DATASET_PASCAL3D:
            logger.info('PASCAL3D')
            metric_df[TABLE_CATEGORIES_PASCAL3D[-1]] = [metric_df[TABLE_CATEGORIES_PASCAL3D[:-1]].loc['mean'].mean(),
                                                        metric_df[TABLE_CATEGORIES_PASCAL3D[:-1]].loc['std'].mean()]

            logger.info(tabulate(metric_df[TABLE_CATEGORIES_PASCAL3D], headers='keys', tablefmt='latex', floatfmt=".2f"))
        elif metrics_dataset[m] == DATASET_OBJECTNET3D:
            logger.info('ObjectNet3D')
            metric_df[TABLE_CATEGORIES_OBJECTNET3D_3[-1]] = [metric_df[TABLE_CATEGORIES_OBJECTNET3D_3[:-1]].loc['mean'].mean(),
                                                             metric_df[TABLE_CATEGORIES_OBJECTNET3D_3[:-1]].loc['std'].mean()]

            logger.info(tabulate(metric_df[TABLE_CATEGORIES_OBJECTNET3D_3], headers='keys', tablefmt='latex', floatfmt=".2f"))

            metric_df[TABLE_CATEGORIES_OBJECTNET3D_23[-1]] = [metric_df[TABLE_CATEGORIES_OBJECTNET3D_23[:-1]].loc['mean'].mean(),
                                                                metric_df[TABLE_CATEGORIES_OBJECTNET3D_23[:-1]].loc['std'].mean()]

            logger.info(tabulate(metric_df[TABLE_CATEGORIES_OBJECTNET3D_23], headers='keys', tablefmt='latex', floatfmt=".2f"))

        elif metrics_dataset[m] == DATASET_CO3D_20:
            logger.info('CO3D 20')

            metric_df[TABLE_CATEGORIES_CO3D_20[-1]] = [metric_df[TABLE_CATEGORIES_CO3D_20[:-1]].loc['mean'].mean(),
                                                       metric_df[TABLE_CATEGORIES_CO3D_20[:-1]].loc['std'].mean()]
            logger.info('Dataset: ZSP, Categories: ZSP')
            logger.info(tabulate(metric_df[TABLE_CATEGORIES_ZSP], headers='keys', tablefmt='latex', floatfmt=".2f"))
            logger.info('Dataset: ZSP, Categories: YOLO')
            logger.info(tabulate(metric_df[TABLE_CATEGORIES_YOLO], headers='keys', tablefmt='latex', floatfmt=".2f"))
            logger.info('Dataset: ZSP, Categories: 20')
            logger.info(tabulate(metric_df[TABLE_CATEGORIES_CO3D_20], headers='keys', tablefmt='latex', floatfmt=".2f"))

        elif metrics_dataset[m] == DATASET_CO3D_28:
            metric_df[TABLE_CATEGORIES_CO3D_20[-1]] = [metric_df[TABLE_CATEGORIES_CO3D_20[:-1]].loc['mean'].mean(),
                                                       metric_df[TABLE_CATEGORIES_CO3D_20[:-1]].loc['std'].mean()]
            metric_df[TABLE_CATEGORIES_CO3D_28[-1]] = [metric_df[TABLE_CATEGORIES_CO3D_28[:-1]].loc['mean'].mean(),
                                                       metric_df[TABLE_CATEGORIES_CO3D_28[:-1]].loc['std'].mean()]

            logger.info('Dataset: Ours, Categories: ZSP')
            logger.info(tabulate(metric_df[TABLE_CATEGORIES_ZSP], headers='keys', tablefmt='latex', floatfmt=".2f"))
            logger.info('Dataset: Ours, Categories: YOLO')
            logger.info(tabulate(metric_df[TABLE_CATEGORIES_YOLO], headers='keys', tablefmt='latex', floatfmt=".2f"))
            logger.info('Dataset: Ours, Categories: 28')
            logger.info(tabulate(metric_df[TABLE_CATEGORIES_CO3D_28], headers='keys', tablefmt='latex', floatfmt=".2f"))

        else:
            #logger.info(metric_df.to_latex(float_format=".2f"))
            logger.info(tabulate(metric_df, headers='keys', tablefmt='latex', floatfmt=".2f"))

@app.command()
def pose_pi6_categories_align3d_co3dv1():
    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D

    # CO3Dv1_NeMo, metrics
    age_in_hours = 24
    configs = []
    #  ZSP v1
    # CO3Dv1_NeMo, metrics
    name_regex = '10-29_00-20-19_CO3Dv1_NeMo_Align3D_local' # 10-29_00-20-19_CO3Dv1_NeMo_Align3D_local
    columns_map = {}
    metrics = [] #  ['pose/acc_pi6', 'pose/acc_pi6_std']
    columns_std = []
    columns_mean = []
    columns_mean_map = {}
    columns_std_map = {}
    for category in TABLE_CATEGORIES_CO3D_20[:-1]:
        metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        columns_map[metrics[-1]] = category + '_mean'
        columns_mean.append(columns_map[metrics[-1]])
        columns_mean_map[columns_map[metrics[-1]]] = category
        metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6_std')
        columns_map[metrics[-1]] = category + '_std'
        columns_std.append(columns_map[metrics[-1]])
        columns_std_map[columns_map[metrics[-1]]] = category

    columns_mean_map = dict(zip(columns_mean, columns_map.values()))
    columns_std_map = dict(zip(columns_std, columns_map.values()))

    df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours, name_regex=name_regex)
    df = df.rename(columns=columns_map)
    metric_df = pd.concat([df[columns_mean].rename(columns=columns_mean_map), df[columns_std].rename(columns=columns_std_map)])
    metric_df = metric_df * 100.
    metric_df['category'] = ['mean', 'std']
    return metric_df

    #metric_df= metric_df.set_index('category')
    #metric_df = pd.concat([df[columns_mean].rename(columns=columns_mean_map), df[columns_std].rename(columns=columns_std_map)])


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
    #  ZSP v1
    # CO3Dv1_NeMo, metrics
    pascal3d_nemo_name_partial = '10-29_00-20-19_CO3Dv1_NeMo_Align3D_local' # 10-29_00-20-19_CO3Dv1_NeMo_Align3D_local

    pascal3d_nemo_metrics = []
    pascal3d_nemo_columns_map = {}
    #pascal3d_nemo_metrics = ['test/pascal3d_test/pose/acc_pi6']
    #pascal3d_nemo_columns_map[pascal3d_nemo_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        pascal3d_nemo_metrics.append(f'test/pascal3d_test/pose/prefix/{category}_acc_pi6')
        pascal3d_nemo_columns_map[pascal3d_nemo_metrics[-1]] = category
    pascal3d_nemo_df = get_dataframe(configs=configs, metrics=pascal3d_nemo_metrics, age_in_hours=age_in_hours, name_regex=pascal3d_nemo_name_partial)
    pascal3d_nemo_df = pascal3d_nemo_df.rename(columns=pascal3d_nemo_columns_map)




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
