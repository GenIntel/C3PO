import pandas
import torchvision.io
import typer
import od3d.io
from omegaconf import OmegaConf
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from od3d.benchmark.run import bench_single_method_local, bench_single_method_local_separate_venv, bench_single_method_local_docker, bench_single_method_torque, bench_single_method_slurm
from od3d.benchmark.benchmark import get_timestamp_as_string, get_timestamp_from_string
import json
app = typer.Typer()
import subprocess
from omegaconf import open_dict
import time

import datetime
import pandas as pd
from pygit2 import Repository

from tabulate import tabulate
import re
from od3d.datasets.frame import OD3D_Meta
import od3d.io

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_nested_value(data, key):
    keys = key.split('.')  # Split the string key into a list of keys
    value = data
    for k in keys:
        if k in value:
            value = value[k]
        elif 'value' in value and k in value['value']:
            value = value['value'][k]
        else:
            return None  # Key not found
    return value

def get_dataframe(configs=[], metrics=[], name_partial=None, age_in_hours=None, name_partial_ban=None):
    logging.basicConfig(level=logging.INFO)
    import wandb
    config = od3d.io.load_hierarchical_config()

    # Initialize wandb
     # wandb.init(project=config.logger.wandb_project_name)

    # Access the API
    api = wandb.Api()


    # config.logger.wandb_project_name
    # Fetch all the runs in your project
    runs = api.runs(config.logger.wandb_project_name)

    if age_in_hours is not None:
        runs = list(filter(
            lambda run: get_timestamp_from_string(run.name) > datetime.datetime.now() - datetime.timedelta(hours=age_in_hours),
            runs))

    if name_partial is not None:
        runs = list(filter(lambda run: name_partial in run.name, runs))

    if name_partial_ban is not None:
        for n in name_partial_ban:
            runs = list(filter(lambda run: n not in run.name, runs))

    if metrics is not None:
        runs = list(filter(lambda run: all([metric in list(run.summary.keys()) for metric in metrics]), runs))


    rows = []
    for run in runs:
        try:
            logger.info(f'runs {run.name}')
            run_summary = run.summary
            row = [run.name]
            if len(configs) > 0:
                json_config = json.loads(run.json_config)
            for config in configs:
                row.append(get_nested_value(json_config, key=config))
            for metric in metrics:
                row.append(run_summary[metric])

            rows.append(row)
        except Exception as e:

            logger.warning(f'skipping run {run.name} due to {e}')
    logger.info(rows)

    #cols = ['name'] + metrics

    cols = ['name'] + configs + metrics

    df = pd.DataFrame(rows, columns=cols)

    #logger.info(tabulate(rows, headers=cols, tablefmt='github',  floatfmt=".3f")) # 'github', 'tsv'
    #logger.info(tabulate(rows, headers=cols, tablefmt='html',  floatfmt=".3f")) # 'github', 'tsv'

    cols_renames = {
        'name': "Run",
        'test/pascal3d_test/pose/acc_pi6': "Acc. Pi/6. [%]",
        'test/pascal3d_test/pose/acc_pi18': "Acc. Pi/18. [%]",
        'test/pascal3d_test/pose/err_median': "Median [deg.]",
        'test/pascal3d_test/pose/err_mean': "Mean [deg.]",
        'test/pascal3d_test/time_pose': 'Inference Duration [s]',
        'test/co3d_5s_test/pose/acc_pi6': "Acc. Pi/6. [%]",
        'test/co3d_5s_test/pose/acc_pi18': "Acc. Pi/18. [%]",
        'test/co3d_5s_test/pose/err_median': "Median [deg.]",
        'test/co3d_5s_test/pose/err_mean': "Mean [deg.]",
        'test/co3d_5s_test/time_pose': 'Duration [s]',
        'test/co3d_50s_test/pose/acc_pi6': "Acc. Pi/6. [%]",
        'test/co3d_50s_test/pose/acc_pi18': "Acc. Pi/18. [%]",
        'test/co3d_50s_test/pose/err_median': "Median [deg.]",
        'test/co3d_50s_test/pose/err_mean': "Mean [deg.]",
        'test/co3d_50s_test/time_pose': 'Inference Duration [s]',
        'test/co3d/pose/acc_pi6': "Acc. Pi/6. [%]",
        'test/co3d/pose/acc_pi18': "Acc. Pi/18. [%]",
        'test/co3d/pose/err_median': "Median [deg.]",
        'test/co3d/pose/err_mean': "Mean [deg.]",
        'test/co3d/time_pose': 'Inference Duration [s]',
    }

    cols_scales = {
        'test/pascal3d_test/pose/acc_pi6': 100.,
        'test/pascal3d_test/pose/acc_pi18': 100.,
        'test/co3d_5s_test/pose/acc_pi6': 100.,
        'test/co3d_5s_test/pose/acc_pi18': 100.,
        'test/co3d_50s_test/pose/acc_pi6': 100.,
        'test/co3d_50s_test/pose/acc_pi18': 100.,
        'test/co3d/pose/acc_pi6': 100.,
        'test/co3d/pose/acc_pi18': 100.,
    }

    for col in cols_scales.keys():
        if col in df:
            df[col] = df[col] * cols_scales[col]
    df = df.rename(columns=cols_renames)


    return df
@app.command()
def table():
    logging.basicConfig(level=logging.INFO)
    #import wandb
    # config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm



    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean', 'test/pascal3d_test/time_pose']
    metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean', 'test/co3d_5s_test/time_pose']

    #metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean', 'test/co3d_50s_test/time_pose']
    name_partial = 'no_early' # None, 'inference', 'split', 'render'
    # configs = ['method.value.multiview.type', 'method.value.multiview.batch_size']
    age_in_hours = 250
    configs = []

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours, name_partial=name_partial)


    # logger.info(tabulate(my_df, headers='keys', tablefmt='tsv',  floatfmt=".3f")) # 'github', 'tsv'
    # logger.info('\n' + my_df.to_csv(sep='\t', index=False, float_format="%.3f"))
    logger.info('\n' + my_df.to_csv(sep=',', index=False, float_format="%.3f"))


    # my_df.to_csv('output.csv', index=False, header=False, float_format='%.3f')

@app.command()
def table_multiple_categories_multiview_incremental():

    logging.basicConfig(level=logging.INFO)

    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    #metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    metrics = ['test/co3d/pose/acc_pi6', 'test/co3d/pose/acc_pi18',
               'test/co3d/pose/err_median', 'test/co3d/pose/err_mean']

    name_partial = '_1s_' # _1s_ 'multiview' _mv6_
    name_partial_ban = ['45s']
    configs = ['method.class_name', 'train_datasets.labeled.categories', 'method.value.multiview.type', 'method.value.multiview.batch_size', 'train_datasets.labeled.dict_nested_frames']
    age_in_hours = 24 * 10

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours, name_partial=name_partial, name_partial_ban=name_partial_ban)

    my_df['sequence_nth'] = my_df["Run"].str.split('_1s_').str[1:2].str.join('_').str[:3]
    my_df['train_datasets.labeled.categories'] = my_df['train_datasets.labeled.categories'].str[0]
    my_df = my_df.groupby(['train_datasets.labeled.categories', 'method.class_name']).head(3)

    # NeMo, NeMo_MultiView, NeMo_Incremental

    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    map_columns = {
        'train_datasets.labeled.categories': 'category',
        'method.class_name': 'method',
        metrics[0]: metrics_new_names[0],
        metrics[1]: metrics_new_names[1],
        metrics[2]: metrics_new_names[2],
        metrics[3]: metrics_new_names[3]
    }

    my_df = my_df.rename(columns=map_columns)
    my_df = my_df.sort_values(by=['category', 'method', 'sequence_nth'])
    my_df = my_df.reset_index(drop=True)

    for i, metric in enumerate(metrics_new_names):
        #my_df = my_df.sort_values(by=['category', metric])

        mv_plot = sns.catplot(
            x="category",  # x variable name
            y=metric,  # y variable name
            hue="method",  # group variable name
            data=my_df,  # dataframe to plot
            kind="bar",
            errorbar=('ci', 100)
        )

        plt.savefig(f"method_cats_{metrics[i].replace('/', '_')}.png")
        plt.clf()



    my_df.to_csv('output.csv', index=False, header=False)

def save_category_sequence_images_as_one(df: pandas.DataFrame):
    # df has to contain 'train_datasets.labeled.dict_nested_frames', 'category', 'sequence_nth'
    config = od3d.io.load_hierarchical_config()

    def get_first_non_none_value(dictionary):
        if dictionary is None:
            return None
        for key, value in dictionary.items():
            if value is not None:
                return f'{key}/{list(value.keys())[0]}'
        return None

    sequences_name = [get_first_non_none_value(config) for config in df['train_datasets.labeled.dict_nested_frames']]
    sequences_path_imgs = [Path(config.platform_local.path_datasets).joinpath('CO3D', sequence_name, 'images') if sequence_name is not None else None for sequence_name in sequences_name]
    sequences_fpath_first_imgs = [sorted(list(sequence_path_imgs.iterdir()), key=lambda f: [OD3D_Meta.atoi(val) for val in re.split(r'(\d+)', f.stem)]) if sequence_path_imgs is not None else None for sequence_path_imgs in sequences_path_imgs]
    #sequences_fpath_first_img = [sequence_fpath_first_imgs[0] for sequence_fpath_first_imgs in sequences_fpath_first_imgs]
    df['sequence'] = sequences_name
    df['sequences_fpath_first_imgs'] = sequences_fpath_first_imgs

    df_category_sequence_unique = df.groupby(['category', 'sequence_nth']).head(1)
    df_category_sequence_unique = df_category_sequence_unique.reset_index(drop=True)
    from od3d.cv.visual.show import fpaths_to_rgb, show_img, imgs_to_img
    from od3d.cv.visual.draw import draw_text_in_rgb
    import torch
    category_sequence_rgbs = []
    for i, sequence_fpath_first_imgs in enumerate(df_category_sequence_unique['sequences_fpath_first_imgs']):
        ids = np.linspace(0, len(sequence_fpath_first_imgs)-1, 3, dtype=int)
        category_sequence_rgbs.append(draw_text_in_rgb(fpaths_to_rgb(fpaths=[sequence_fpath_first_imgs[id] for id in ids], H=512, W=512), text=f"\n\n\n{df_category_sequence_unique['category'][i]} {df_category_sequence_unique['sequence_nth'][i]}"))
        # , fpath=f"{df_category_sequence_unique['category'][i]}_{df_category_sequence_unique['sequence_nth'][i]}.png"
    show_img(imgs_to_img(torch.stack(category_sequence_rgbs, dim=0)[:, None]), fpath='category_sequences.png')

@app.command()
def table_multiple_categories():
    logging.basicConfig(level=logging.INFO)
    # config = od3d.io.load_hierarchical_config()

    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    #metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    metrics = ['test/co3d/pose/acc_pi6', 'test/co3d/pose/acc_pi18',
               'test/co3d/pose/err_median', 'test/co3d/pose/err_mean']

    name_partial = '_1s_' # _1s_ 'multiview' _mv6_
    name_partial_ban = ['MultiView', 'Incremental']
    configs = ['train_datasets.labeled.categories', 'method.value.multiview.type', 'method.value.multiview.batch_size']
    age_in_hours = 24 * 4

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours, name_partial=name_partial, name_partial_ban=name_partial_ban)

    import seaborn as sns
    import matplotlib.pyplot as plt

    my_df['train_datasets.labeled.categories'] = my_df['train_datasets.labeled.categories'].str[0]
    my_df = my_df.groupby(['train_datasets.labeled.categories']).head(3)

    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    map_columns = {
        'train_datasets.labeled.categories': 'category',
        metrics[0]: metrics_new_names[0],
        metrics[1]: metrics_new_names[1],
        metrics[2]: metrics_new_names[2],
        metrics[3]: metrics_new_names[3]
    }

    my_df = my_df.rename(columns=map_columns)
    my_df = my_df.sort_values(by=['category'])
    import numpy as np

    for i, metric in enumerate(metrics_new_names):
        my_df = my_df.sort_values(by=['category', metric])

        yerr = np.stack([my_df.groupby('category')[metric].min().to_numpy(), my_df.groupby('category')[metric].max().to_numpy()])

        mv_plot = sns.barplot(data=my_df, x="category", y=metric, errorbar=('ci', 100))

        #mv_plot = sns.catplot(
        #    x="train seqs.",  # x variable name
        #    y=metric,  # y variable name
        #    hue="inference type",  # group variable name
        #    data=my_df,  # dataframe to plot
        #    kind="bar",
        #)

        plt.savefig(f"single_{metrics[i].replace('/', '_')}.png")
        plt.clf()



    my_df.to_csv('output.csv', index=False, header=False)


@app.command()
def table_multiview_sequences():
    logging.basicConfig(level=logging.INFO)
    # config = od3d.io.load_hierarchical_config()

    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    name_partial = 'multiview'
    configs = ['method.value.multiview.type', 'method.value.multiview.batch_size', 'method.value.inference.refine.dims_detached']
    age_in_hours = 6

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours, name_partial=name_partial)

    import seaborn as sns
    import matplotlib.pyplot as plt

    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    #my_df['seqs'] = my_df["Run"].str.split('s_').str[0:1]
    map_seqs = {
        '_1st_slurm': '1',
        '_2nd_slurm': '1',
        '_3rd_slurm': '1',
        's8_2_slurm': '2',
        's8_3_slurm': '3',
    }
    my_df.loc[my_df['method.value.inference.refine.dims_detached'].map(len) == 0, 'method.value.multiview.type'] = 'multiview+translation'
    my_df['seqs'] = my_df["Run"].str.split('s_').str[0:2].str.join('_').str[-10:].replace(map_seqs)
    my_df = my_df.loc[my_df['seqs'] != '_bs8_slurm']
    #my_df = my_df.groupby(['method.value.multiview.type', 'seqs']).head(1)
    my_df = my_df.rename(columns={'seqs': 'train seqs.', 'method.value.multiview.type': 'inference type', 'method.value.multiview.batch_size': 'multiview #frames', metrics[0]: metrics_new_names[0], metrics[1]: metrics_new_names[1], metrics[2]: metrics_new_names[2], metrics[3]: metrics_new_names[3]})
    my_df = my_df.sort_values(by=['train seqs.', 'inference type'])

    for i, metric in enumerate(metrics_new_names):
        mv_plot = sns.catplot(
            x="train seqs.",  # x variable name
            y=metric,  # y variable name
            hue="inference type",  # group variable name
            data=my_df,  # dataframe to plot
            kind="bar",
        )
        #fig = mv_plot.get_figure()
        plt.savefig(f"multiview_{metrics[i].replace('/', '_')}.png")

    my_df.to_csv('output.csv', index=False, header=False)


@app.command()
def table_multiview():
    logging.basicConfig(level=logging.INFO)
    # config = od3d.io.load_hierarchical_config()

    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    name_partial = 'multiview'
    configs = ['method.value.multiview.type', 'method.value.multiview.batch_size']
    age_in_hours = 2

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours=age_in_hours, name_partial=name_partial)

    import seaborn as sns
    import matplotlib.pyplot as plt

    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    my_df = my_df.groupby(['method.value.multiview.type', 'method.value.multiview.batch_size']).head(1)
    my_df = my_df.rename(columns={'method.value.multiview.type': 'type', 'method.value.multiview.batch_size': 'multiview #frames', metrics[0]: metrics_new_names[0], metrics[1]: metrics_new_names[1], metrics[2]: metrics_new_names[2], metrics[3]: metrics_new_names[3]})
    my_df = my_df.sort_values(by=['type'])

    for i, metric in enumerate(metrics_new_names):
        mv_plot = sns.catplot(
            x="multiview #frames",  # x variable name
            y=metric,  # y variable name
            hue="type",  # group variable name
            data=my_df,  # dataframe to plot
            kind="bar",
        )
        #fig = mv_plot.get_figure()
        plt.savefig(f"multiview_{metrics[i].replace('/', '_')}.png")

    my_df.to_csv('output.csv', index=False, header=False)



@app.command()
def multiple(benchmark: str = typer.Option('co3d_nemo', '-b', '--benchmark'),
        ablation: str = typer.Option(None, '-a', '--ablation'),
        platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)

    file_fpath = Path(__file__).parent.resolve()
    config_dir_rel = "../../../config"
    config_dir_abs = file_fpath.joinpath(config_dir_rel)
    ablations_root_dir = config_dir_abs.joinpath("ablations")

    if ablation is None:
        cfgs = [od3d.io.load_hierarchical_config(benchmark=benchmark, platform=platform)]
    else:
        cfgs = []
        # create one config per ablation
        ablation_dir = ablations_root_dir.joinpath(ablation)
        for ablation_file_fpath in ablation_dir.iterdir():
            if ablation_file_fpath.is_dir():
                continue
            ablation_fpath_rel = ablation_file_fpath.relative_to(ablations_root_dir).with_suffix('')
            if not ablation_fpath_rel.name.startswith("_"):
                cfgs.append(od3d.io.load_hierarchical_config(benchmark=benchmark, platform=platform, ablation=str(ablation_fpath_rel)))

    # create one config per method
    methods_cfgs = []
    for cfg in cfgs:
        methods_keys = cfg.method.keys()
        for key in methods_keys:
            method_cfg = cfg.copy()
            method_cfg.method = cfg.method[key]
            method_cfg_exists = False
            for prev_method_cfg in methods_cfgs:
                if method_cfg == prev_method_cfg:
                    method_cfg_exists = True
            if not method_cfg_exists:
                methods_cfgs.append(method_cfg)

    print(f"{len(methods_cfgs)} configs with single method.")

    current_branch = Repository('.').head.shorthand  # 'master'

    for method_cfg in methods_cfgs:
        with open_dict(method_cfg):
            if method_cfg.get('branch', None) is None:
                method_cfg.branch = current_branch

            ablation_name = method_cfg.get("ablation_name", None)
            if ablation_name is not None:
                method_cfg.run_name = f'{get_timestamp_as_string()}_{method_cfg.train_datasets.labeled.class_name}_{method_cfg.method.class_name}_{ablation_name}_{method_cfg.platform.link}'
            else:
                method_cfg.run_name = f'{get_timestamp_as_string()}_{method_cfg.train_datasets.labeled.class_name}_{method_cfg.method.class_name}_{method_cfg.platform.link}'

        if method_cfg.platform.link == 'local':
            bench_single_method_local(method_cfg)
        elif method_cfg.platform.link == 'local-separate-venv':
            bench_single_method_local_separate_venv(method_cfg)
        elif method_cfg.platform.link == 'local-docker':
            bench_single_method_local_docker(method_cfg)
        elif method_cfg.platform.link == 'torque':
            bench_single_method_torque(method_cfg)
        elif method_cfg.platform.link == 'slurm':
            bench_single_method_slurm(method_cfg)

        time.sleep(5)



@app.command()
def single_local(config_fpath: str = typer.Option(None, '-c', '--config')):
    logging.basicConfig(level=logging.INFO)
    method_cfg = OmegaConf.load(config_fpath)
    bench_single_method_local(method_cfg)


@app.command()
def test(benchmark: str = typer.Option('timeseries_internal', '-b', '--benchmark'),
         mode: str = typer.Option('local', '-m', '--mode'),
         tasks: str = typer.Option(None, '-t', '--tasks'),
         constraint: str = typer.Option('4h8c', '-c', '--constraint'),
         frameworks: str = typer.Option(None, '-f', '--frameworks'),
         localcode: str = typer.Option(None, '-l', '--localcode')):
    logging.basicConfig(level=logging.DEBUG)
    logger.info("test")
@app.command()
def info_slurm():
    'scontrol show job'

    'srun -p lmb_gpu-rtx2080 -w dagobert --pty bash'
    pass

def get_slurm_jobs_ids(job_id_treshold=None):
    slurm_result = subprocess.run(f'ssh slurm "squeue --me"', capture_output=True, shell=True)
    slurm_jobs = slurm_result.stdout.decode("utf-8").split("\n")
    slurm_jobs_ids = []
    for slurm_job in slurm_jobs[1:]:
        slurm_job_split = slurm_job.split()
        if len(slurm_job_split) > 0:
            slurm_jobs_ids.append(int(slurm_job_split[0]))
    if job_id_treshold is not None:
        slurm_jobs_ids = list(filter(lambda job_id: job_id < job_id_treshold, slurm_jobs_ids))
    return slurm_jobs_ids

@app.command()
def rsync(platform_source: str = typer.Option('slurm', '-s', '--source'),
          platform_target: str = typer.Option('local', '-t', '--target'),
          run: str = typer.Option(None, '-r', '--run')):
    logging.basicConfig(level=logging.INFO)
    if run is None:
        logger.warning('Please specify a run.')
        return

    config_source = od3d.io.load_hierarchical_config(platform=platform_source)
    config_target = od3d.io.load_hierarchical_config(platform=platform_target)
    source_link = f'{config_source.platform.link}:' if config_source.platform.link != 'local' else ''
    target_link = f'{config_target.platform.link}:' if config_target.platform.link != 'local' else ''

    path_source = Path(config_source.platform.path_exps).joinpath(run)
    path_target = Path(config_target.platform.path_exps).joinpath(run)
    od3d.io.run_cmd(cmd=f'rsync -avrzP {source_link}{path_source} {target_link}{path_target.parent}', live=True, logger=logger)

@app.command()
def status_slurm():
    logging.basicConfig(level=logging.INFO)
    # 60j = 60 characters
    format = '"%.18i %.9P %.60j %.8u %.8T %.10M %.9l %.6D %R"'
    slurm_result = subprocess.run(f"ssh slurm 'squeue --me --format={format}'", capture_output=True, shell=True)
    slurm_jobs = slurm_result.stdout.decode("utf-8").split("\n")
    for slurm_job in slurm_jobs:
        logger.info(slurm_job)
@app.command()
def status_torque():
    logging.basicConfig(level=logging.INFO)

    torque_result = subprocess.run(f'ssh torque "qstat -a -u $(whoami)"', capture_output=True, shell=True)
    torque_jobs = torque_result.stdout.decode("utf-8").split("\n")
    for torque_job in torque_jobs:
        logger.info(torque_job)
@app.command()
def stop_torque(job: str = typer.Option(None, '-j', '--job')):
    logging.basicConfig(level=logging.INFO)

    jobs = job.split(',')
    for job in jobs:
        torque_result = subprocess.run(f'ssh torque "qdel {job}"', capture_output=True, shell=True)
        for line in torque_result.stdout.decode("utf-8").split("\n"):
            logger.info(line)

@app.command()
def stop_slurm(job: str = typer.Option(None, '-j', '--job')):
    logging.basicConfig(level=logging.INFO)

    jobs = job.split(',')
    for job in jobs:
        if job.startswith('l'):
            slurm_jobs_ids = get_slurm_jobs_ids(int(job[1:]))
            logger.info(f'stop slurm job ids {slurm_jobs_ids}')
        else:
            slurm_jobs_ids = [int(job)]

        for job_id in slurm_jobs_ids:
            slurm_result = subprocess.run(f'ssh slurm "scancel {str(job_id)}"', capture_output=True, shell=True)
            for line in slurm_result.stdout.decode("utf-8").split("\n"):
                logger.info(line)