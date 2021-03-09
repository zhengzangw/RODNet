import argparse
import os
import pickle
import time

import numpy as np
from cruw import CRUW
from cruw.mapping import idx2ra, ra2idx
from rodnet.core.object_class import get_class_name
from rodnet.core.post_processing import (
    post_process_single_frame,
    write_dets_results_single_frame,
)
from rodnet.utils.load_configs import load_configs_from_file
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--data_root", type=str, default="rod2021_valid")
    args = parser.parse_args()
    return args


def process(confmap):
    if len(confmap) == 3:
        return confmap
    elif len(confmap) == 4:
        return confmap[1:]
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    confmaps = pickle.load(open(args.input, "rb"))
    
    # cnf = confmaps['2019_04_09_PMS1000'][0]
    # breakpoint()

    out_dir = args.output
    os.makedirs(out_dir)

    dataset = CRUW(
        data_root=args.data_root, sensor_config_name="sensor_config_rod2021",
    )
    config_dict = load_configs_from_file(args.config)

    for seq in tqdm(confmaps, desc="seq", leave=False):
        out_txt = os.path.join(out_dir, f"{seq}.txt")
        f_out = open(out_txt, "w")
        for frame_id in tqdm(confmaps[seq], desc="frame", leave=False):
            # input_confmap should be [3, 128, 128]
            input_confmap = process(confmaps[seq][frame_id])
            assert input_confmap.shape == (3, 128, 128)
            res_final = post_process_single_frame(input_confmap, dataset, config_dict)
            # breakpoint()
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
                print(res, file=f_out)
        f_out.close()
