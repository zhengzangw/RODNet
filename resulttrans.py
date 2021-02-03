import argparse
import os
import pathlib
import shutil
from itertools import *

from cruw import CRUW
from cruw.mapping import idx2ra, ra2idx

data_root = "./ROD2021"
dataset = CRUW(data_root=data_root, sensor_config_name="sensor_config_rod2021")

val_seqs = [
    "2019_04_09_PMS1000",
    "2019_04_30_MLMS001",
    "2019_04_30_PM2S003",
    "2019_05_09_PBMS004",
    "2019_05_23_PM1S015",
    "2019_05_29_MLMS006",
    "2019_09_29_ONRD001",
    "2019_09_29_ONRD013",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-i", "--input", type=str)

    parser.add_argument("--no_transform", action="store_true")
    parser.add_argument("--seq_split", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("% This is output result transformer to fit ROD2021 competition")

    args = parse_args()
    in_dir = args.input
    out_dir = args.output
    assert os.path.exists(in_dir)

    if not args.no_transform:
        os.makedirs(out_dir)
        print(f"> Create {out_dir}")

        for seq in val_seqs:
            result_txt = os.path.join(in_dir, seq, "rod_res.txt")
            out_txt = os.path.join(out_dir, f"{seq}.txt")
            f_in = open(result_txt, "r")
            f_out = open(out_txt, "w")

            for line in f_in:
                f_id, class_name, row_id, col_id, conf = line.split()
                conf = float(conf)
                if conf > 1:
                    conf = 1
                rng, azm = idx2ra(
                    int(row_id), int(col_id), dataset.range_grid, dataset.angle_grid
                )
                res = f"{f_id} {rng} {azm} {class_name} {conf}"
                print(res, file=f_out)

            f_in.close()
            f_out.close()

    if args.seq_split:
        for src_seq in val_seqs:
            src_txt = os.path.join(in_dir, f"{src_seq}.txt")
            assert os.path.exists(src_txt)

            tgt_dir = out_dir + "_" + src_seq
            os.makedirs(tgt_dir)
            tgt_txt = os.path.join(tgt_dir, f"{src_seq}.txt")
            shutil.copy(src_txt, tgt_txt)
            print(f"> Create {tgt_dir}")

    print("> Transform completed!")
