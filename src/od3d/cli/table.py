import logging
logger = logging.getLogger(__name__)
import typer
import od3d.io

app = typer.Typer()
from od3d.cli.benchmark import get_dataframe

@app.command()
def pose_pi6_categories():
    logging.basicConfig(level=logging.INFO)
    #import wandb
    categories = od3d.io.read_config_intern('datasets/categories/zsp.yaml')

    #config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm

    # CO3Dv1_NeMo, metrics
    nemo_metrics = []
    for category in categories:
        nemo_metrics.append(f'val/co3d_10s_zsp_test/pose/prefix/{category}_acc_pi6')
    'val/co3d_10s_zsp_test/pose/prefix/toilet_acc_pi6'
    'val/co3d_10s_zsp_test/pose/acc_pi6'

    # CO3Dv1_NeMo_Align3D, metrics
    'only_to_ref/pose/prefix/book_acc_pi6'
    'pose/prefix/handbag_acc_pi6'
    'pose/acc_pi6'

    metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean', 'test/pascal3d_test/time_pose']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean', 'test/co3d_5s_test/time_pose']

    #metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean', 'test/co3d_50s_test/time_pose']
    name_partial = '_car_1s_' # None, 'inference', 'split', 'render'
    # configs = ['method.value.multiview.type', 'method.value.multiview.batch_size']
    age_in_hours = 250
    configs = []

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours, name_partial=name_partial)


    # logger.info(tabulate(my_df, headers='keys', tablefmt='tsv',  floatfmt=".3f")) # 'github', 'tsv'
    # logger.info('\n' + my_df.to_csv(sep='\t', index=False, float_format="%.3f"))
    logger.info('\n' + my_df.to_csv(sep=',', index=False, float_format="%.3f"))
