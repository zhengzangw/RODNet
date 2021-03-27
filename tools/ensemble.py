import argparse
import os
import pathlib
import shutil
from itertools import *

import numpy as np
from cruw import CRUW
from cruw.annotation.init_json import init_meta_json
from cruw.mapping import idx2ra, ra2idx
from rodnet.core.confidence_map import generate_confmap, normalize_confmap
from rodnet.core.object_class import get_class_name
from rodnet.core.post_processing import post_process_single_frame
from rodnet.utils.load_configs import load_configs_from_file
from tqdm import tqdm

in_dirs = [
    # submits/valid/
    # "submits/valid/adam_wc_resnet_60",  # resnet_v01
    # "submits/valid/adam_wc_resnet_m_50",  # resnet_v02
    # "submits/valid/resnet_awc_3lr5_50",  # resnet_v04
    # "submits/valid/resnet_awc_lr3_40",  # resnet_v05
    # "submits/valid/resnet_awc_lr4_50",  # resnet_v06
    # "submits/valid/resnet_v11_50",  # resnet_v11
    # "submits/valid/resnetb_awc_lr3_20",  # resnetb_v01
    # "submits/valid/resnetb_awc_lr4_50",  # resnetb_v02
    # "submits/valid/resnetb_v05_30",  # resnetb_v05
    # "submits/valid/resnetc_v01_50",  # resnetc_v01
    # "submits/valid/c21d_lr5_40",  # c21d_v01
    # "submits/valid/c21d_lr5_40",  # c21d_v02
    # "submits/valid/c21d_16",  # c21d_v03
    # "submits/valid/c21d_v04_18",  # c21d_v04
    # "submits/valid/hgwi_43",
    # "submits/valid/hg_lr4_15",
    # "submits/valid/hg_50",
    # "submits/valid/cdc_78",
    # "submits/valid/cdc_deep_lr4_20",
    # "submits/valid/cdc_lr4_b_16",
    # "submits/valid/cdc_noise_100",
    # submits/submit/
    "submits/submit/resnet_wcm_lr4_50",
    "submits/submit/resnet_wcm_50",
    "submits/submit/s08_50",
    "submits/submit/awc_resnetb_lr4_65",
    "submits/submit/cdc_deep_37",
    "submits/submit/awc_c21d_lr4_70",
    "submits/submit/hg_50",
    "submits/submit/c21d_30",
    ## Finetune
    "submits/submit/c21d_s05_29",
    "submits/submit/cdc_67",
    "submits/submit/resnetb_s07_35",
    "submits/submit/resnetc_s01_51",
    ### Dynamic
    # "submits/submit/c21d_s05_ftD_31",
    # "submits/submit/cdc_ftD_31",
    # "submits/submit/s06_ftD_30",
    # "submits/submit/s07_ftD_31"
    ### Static
    # "submits/submit/c21d_ftS_31",
    # "submits/submit/cdc_ftS_31",
    # "submits/submit/s06_ftS_30",
    # "submits/submit/s07_ftS_31",
]
data_root = "./rod2021"
dataset = CRUW(data_root=data_root, sensor_config_name="sensor_config_rod2021")
n_class = dataset.object_cfg.n_class
radar_configs = dataset.sensor_cfg.radar_cfg
config_dict = load_configs_from_file("configs/new/ensemble.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str)

    args = parser.parse_args()
    return args


def load_anno_txt(txt_path, n_frame, dataset):
    folder_name_dict = dict(cam_0="IMAGES_0", rad_h="RADAR_RA_H")
    anno_dict = init_meta_json(n_frame, folder_name_dict)
    with open(txt_path, "r") as f:
        data = f.readlines()
    for line in data:
        frame_id, r, a, class_name, conf = line.rstrip().split()
        frame_id = int(frame_id)
        r = float(r)
        a = float(a)
        rid, aid = ra2idx(r, a, dataset.range_grid, dataset.angle_grid)
        anno_dict[frame_id]["rad_h"]["n_objects"] += 1
        anno_dict[frame_id]["rad_h"]["obj_info"]["categories"].append(class_name)
        anno_dict[frame_id]["rad_h"]["obj_info"]["centers"].append([r, a])
        anno_dict[frame_id]["rad_h"]["obj_info"]["center_ids"].append([rid, aid])
        anno_dict[frame_id]["rad_h"]["obj_info"]["scores"].append(conf)

    return anno_dict


def get_confmap(metadata_dict, n_class):
    n_obj = metadata_dict["rad_h"]["n_objects"]
    obj_info = metadata_dict["rad_h"]["obj_info"]
    if n_obj == 0:
        confmap_gt = np.zeros(
            (n_class, radar_configs["ramap_rsize"], radar_configs["ramap_asize"],),
            dtype=float,
        )
    else:
        confmap_gt = generate_confmap(n_obj, obj_info, dataset, config_dict)
        confmap_gt = normalize_confmap(confmap_gt)
    assert confmap_gt.shape == (
        n_class,
        radar_configs["ramap_rsize"],
        radar_configs["ramap_asize"],
    )
    return confmap_gt


def init_confmap(n_frames):
    conf_map = np.zeros(
        (n_class, radar_configs["ramap_rsize"], radar_configs["ramap_asize"],),
        dtype=float,
    )
    return conf_map


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.output
    assert not os.path.exists(out_dir)

    seqs = os.listdir(in_dirs[0])
    for d in in_dirs:
        cur_seqs = os.listdir(d)
        assert seqs == cur_seqs, f"Missing seqs in {d}"
    n_model = len(in_dirs)
    print(f"Ensemble on {n_model} models")
    os.makedirs(out_dir)
    print(f"Create {os.path.abspath(out_dir)}")

    for seq_txt in tqdm(seqs, desc="Seq", leave=False):
        # n_frames
        n_frames = 0
        for d in in_dirs:
            in_txt = os.path.join(d, seq_txt)
            with open(in_txt, "r") as fin:
                try:
                    frame_id, r, a, class_name, conf = (
                        fin.readlines()[-1].rstrip().split()
                    )
                    n_frames = max(n_frames, int(frame_id))
                except:
                    print(f"Error when process {d}")
        n_frames += 1
        # anno_dict
        anno_dicts = []
        for d in in_dirs:
            in_txt = os.path.join(d, seq_txt)
            anno_dict = load_anno_txt(in_txt, n_frames, dataset)
            anno_dicts.append(anno_dict)

        txt = os.path.join(out_dir, seq_txt)
        with open(txt, "w") as f:
            for frame_id in tqdm(range(n_frames), desc="Frame", leave=False):
                conf_maps = init_confmap(n_frames)
                # ensemble
                for anno_dict in anno_dicts:
                    confmaps = get_confmap(anno_dict[frame_id], n_class)
                    conf_maps += confmaps / n_model

                res_final = post_process_single_frame(conf_maps, dataset, config_dict)

                for line in res_final:
                    class_id, row_id, col_id, conf = line
                    class_id = int(class_id)
                    if class_id == -1:
                        continue
                    class_name = get_class_name(class_id, dataset.object_cfg.classes)
                    rng, azm = idx2ra(
                        int(row_id), int(col_id), dataset.range_grid, dataset.angle_grid
                    )
                    res = f"{frame_id} {rng} {azm} {class_name} {conf}"
                    print(res, file=f)
    print(f"Saved to {out_dir}")
