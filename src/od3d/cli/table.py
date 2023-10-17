import logging
logger = logging.getLogger(__name__)
import typer
import od3d.io
import pandas as pd

app = typer.Typer()
from od3d.cli.benchmark import get_dataframe
from tabulate import tabulate

@app.command()
def pose_pi6_categories():
    logging.basicConfig(level=logging.INFO)
    #import wandb
    categories = od3d.io.read_config_intern('datasets/categories/zsp.yaml')

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

    df = pd.concat([nemo_df, align3d_df, align3d_1on1_df])
    cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster']
    df = df[cols]
    logger.info(tabulate(df, headers='keys', tablefmt='latex',  floatfmt=".3f")) # 'github', 'tsv', 'latex', 'latex_raw'
    # logger.info('\n' + my_df.to_csv(sep='\t', index=False, float_format="%.3f"))
    #logger.info('\n' + my_df.to_csv(sep=',', index=False, float_format="%.3f"))
