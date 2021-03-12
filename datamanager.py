import argparse
import os
import pathlib
import shutil
from itertools import *

import numpy as np
from PIL import Image


def create_empty_img(path):
    im = Image.fromarray(np.zeros((1, 1)) + 255).convert("RGB")
    im.save(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--build", type=str)
    parser.add_argument(
        "--type", choices=["empty", "submit", "valid", "valid1"], default="empty", type=str
    )

    args = parser.parse_args()
    return args


data_loc = {
    "test_rad_data": "TEST_RAD_H",
    "train_cam_data": "TRAIN_CAM_0",
    "train_rad_data": "TRAIN_RAD_H",
    "train_rad_label": "TRAIN_RAD_H_ANNO",
    "cam_config": "calib",
}
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
val1_seqs = [
    "2019_05_29_MLMS006",
    "2019_09_29_ONRD013",
]


def link_seq(
    seqs, src_folder="train", tgt_folder="train", load_image=True, load_anno=True
):
    for seq in seqs:
        # target
        seq_dir = os.path.join(root, "sequences", tgt_folder, seq)
        os.makedirs(seq_dir)
        image_dir = os.path.join(seq_dir, "IMAGES_0")
        radar_dir = os.path.join(seq_dir, "RADAR_RA_H")
        anno_txt = os.path.join(root, "annotations", tgt_folder, f"{seq}.txt")

        ## radar
        src = os.path.join(data_loc[f"{src_folder}_rad_data"], seq, "RADAR_RA_H")
        os.symlink(os.path.abspath(src), radar_dir)

        ## image
        if load_image:
            src = os.path.join(data_loc[f"{src_folder}_cam_data"], seq, "IMAGES_0")
            os.symlink(os.path.abspath(src), image_dir)
        else:
            os.makedirs(image_dir)
            chirps = os.listdir(src)
            frames = sorted(list(set([int(chirp.split("_")[0]) for chirp in chirps])))
            imgs_name = [f"{frame:010}.jpg" for frame in frames]
            for img_name in imgs_name:
                create_empty_img(os.path.join(image_dir, img_name))

        # anno
        if load_anno:
            src = os.path.join(data_loc[f"{src_folder}_rad_label"], f"{seq}.txt")
            shutil.copy(src, anno_txt)


if __name__ == "__main__":
    # introduction
    print("% This is dataset manager for ROD2021 (unofficial) #####")
    print(
        f"> This file should be in the same directory with {','.join(data_loc.values())}"
    )
    for d in data_loc.values():
        assert os.path.exists(d)

    args = parse_args()

    rod2021_train_seqs = os.listdir(data_loc["train_rad_data"])
    rod2021_test_seqs = os.listdir(data_loc["test_rad_data"])

    if args.info:
        train_seqs_str = "\n\t".join(rod2021_train_seqs)
        print(f"Training has sequence: \n\t{train_seqs_str}")
        test_seqs_str = "\n\t".join(rod2021_test_seqs)
        print(f"Testing has sequence: \n\t{test_seqs_str}")

    if args.build:
        root = args.build
        assert not os.path.exists(root), f'Directory "{root}" already exists'
        # make dirs
        for d1, d2 in product(["sequences", "annotations"], ["train", "test"]):
            os.makedirs(os.path.join(root, d1, d2))
        # link cam
        os.symlink(
            os.path.abspath(data_loc["cam_config"]),
            os.path.join(root, data_loc["cam_config"]),
        )
        print(f"> Build rod2021 structure {root}")

    print(f"> dataset type: {args.type}")
    if args.type == "submit":
        train_seqs = rod2021_train_seqs
        test_seqs = rod2021_test_seqs
        link_seq(train_seqs, "train", "train")
        link_seq(test_seqs, "test", "test", False, False)
    elif args.type == "valid":
        train_seqs = [seq for seq in rod2021_train_seqs if seq not in val_seqs]
        test_seqs = val_seqs
        link_seq(train_seqs, "train", "train")
        link_seq(test_seqs, "train", "test")
    elif args.type == "valid1":
        train_seqs = [seq for seq in rod2021_train_seqs if seq not in val1_seqs]
        test_seqs = val1_seqs
        link_seq(train_seqs, "train", "train")
        link_seq(test_seqs, "train", "test")

    print("> Completed!")
