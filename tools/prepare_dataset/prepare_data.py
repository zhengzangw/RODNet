import argparse
import json
import os
import pickle
import shutil
import sys

import numpy as np
from cruw import CRUW
from cruw.annotation.init_json import init_meta_json
from cruw.mapping import ra2idx
from rodnet.core.confidence_map import (add_noise_channel, generate_confmap,
                                        normalize_confmap)
from rodnet.utils.load_configs import load_configs_from_file
from rodnet.utils.visualization import visualize_confmap

SPLITS_LIST = ["train", "valid", "test", "demo"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RODNet data.")
    parser.add_argument(
        "--config", type=str, dest="config", help="configuration file path"
    )
    parser.add_argument("--data_root", type=str, help="directory to the prepared data")
    parser.add_argument(
        "--split",
        type=str,
        dest="split",
        help="choose from train, valid, test, supertest",
    )
    parser.add_argument(
        "--out_data_dir",
        type=str,
        default="./data",
        help="data directory to save the prepared data",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite prepared data if exist"
    )
    args = parser.parse_args()
    return args


def load_anno_txt(txt_path, n_frame, dataset):
    folder_name_dict = dict(cam_0="IMAGES_0", rad_h="RADAR_RA_H")
    anno_dict = init_meta_json(n_frame, folder_name_dict)
    with open(txt_path, "r") as f:
        data = f.readlines()
    for line in data:
        frame_id, r, a, class_name = line.rstrip().split()
        frame_id = int(frame_id)
        r = float(r)
        a = float(a)
        rid, aid = ra2idx(r, a, dataset.range_grid, dataset.angle_grid)
        anno_dict[frame_id]["rad_h"]["n_objects"] += 1
        anno_dict[frame_id]["rad_h"]["obj_info"]["categories"].append(class_name)
        anno_dict[frame_id]["rad_h"]["obj_info"]["centers"].append([r, a])
        anno_dict[frame_id]["rad_h"]["obj_info"]["center_ids"].append([rid, aid])
        anno_dict[frame_id]["rad_h"]["obj_info"]["scores"].append(1.0)

    return anno_dict


def generate_confmaps(metadata_dict, n_class, viz):
    # breakpoint()
    confmaps = []
    for metadata_frame in metadata_dict:
        n_obj = metadata_frame["rad_h"]["n_objects"]
        obj_info = metadata_frame["rad_h"]["obj_info"]
        if n_obj == 0:
            confmap_gt = np.zeros(
                (
                    n_class + 1,
                    radar_configs["ramap_rsize"],
                    radar_configs["ramap_asize"],
                ),
                dtype=float,
            )
            confmap_gt[-1, :, :] = 1.0  # initialize noise channal
        else:
            confmap_gt = generate_confmap(n_obj, obj_info, dataset, config_dict)
            confmap_gt = normalize_confmap(confmap_gt)
            confmap_gt = add_noise_channel(confmap_gt, dataset, config_dict)
        assert confmap_gt.shape == (
            n_class + 1,
            radar_configs["ramap_rsize"],
            radar_configs["ramap_asize"],
        )
        if viz:
            visualize_confmap(confmap_gt)
        confmaps.append(confmap_gt)
    confmaps = np.array(confmaps)
    return confmaps


def prepare_data(
    dataset, config_dict, data_dir, split, save_dir, viz=False, overwrite=False
):
    """
    Prepare pickle data for RODNet training and testing
    :param dataset: dataset object
    :param config_dict: rodnet configurations
    :param data_dir: output directory of the processed data
    :param split: train, valid, test, demo, etc.
    :param save_dir: output directory of the prepared data
    :param viz: whether visualize the prepared data
    :param overwrite: whether overwrite the existing prepared data
    :return:
    """
    camera_configs = dataset.sensor_cfg.camera_cfg
    radar_configs = dataset.sensor_cfg.radar_cfg
    n_chirp = radar_configs["n_chirps"]
    n_class = dataset.object_cfg.n_class

    data_root = config_dict["dataset_cfg"]["data_root"]
    anno_root = config_dict["dataset_cfg"]["anno_root"]
    set_cfg = config_dict["dataset_cfg"][split]
    if "seqs" not in set_cfg:
        sets_seqs = sorted(os.listdir(os.path.join(data_root, set_cfg["subdir"])))
    else:
        sets_seqs = set_cfg["seqs"]

    if overwrite:
        if os.path.exists(os.path.join(data_dir, split)):
            shutil.rmtree(os.path.join(data_dir, split))
        os.makedirs(os.path.join(data_dir, split))

    for seq in sets_seqs:
        seq_path = os.path.join(data_root, set_cfg["subdir"], seq)
        seq_anno_path = os.path.join(
            anno_root, set_cfg["subdir"], seq + config_dict["dataset_cfg"]["anno_ext"]
        )
        save_path = os.path.join(save_dir, seq + ".pkl")
        print("Sequence %s saving to %s" % (seq_path, save_path))

        try:
            if not overwrite and os.path.exists(save_path):
                print("%s already exists, skip" % save_path)
                continue

            # Camera
            image_dir = os.path.join(seq_path, camera_configs["image_folder"])
            if os.path.exists(image_dir):
                image_paths = sorted(
                    [
                        os.path.join(image_dir, name)
                        for name in os.listdir(image_dir)
                        if name.endswith(camera_configs["ext"])
                    ]
                )
                n_frame = len(image_paths)
            else:
                print("skip camera images")
                image_paths = []

            # Radar
            radar_dir = os.path.join(
                seq_path, dataset.sensor_cfg.radar_cfg["chirp_folder"]
            )
            if radar_configs["data_type"] == "RI" or radar_configs["data_type"] == "AP":
                radar_paths = sorted(
                    [
                        os.path.join(radar_dir, name)
                        for name in os.listdir(radar_dir)
                        if name.endswith(dataset.sensor_cfg.radar_cfg["ext"])
                    ]
                )
                n_radar_frame = len(radar_paths)
                assert n_frame == n_radar_frame
            elif (
                radar_configs["data_type"] == "RISEP"
                or radar_configs["data_type"] == "APSEP"
            ):
                radar_paths_chirp = []
                for chirp_id in range(n_chirp):
                    chirp_dir = os.path.join(radar_dir, "%04d" % chirp_id)
                    paths = sorted(
                        [
                            os.path.join(chirp_dir, name)
                            for name in os.listdir(chirp_dir)
                            if name.endswith(
                                config_dict["dataset_cfg"]["radar_cfg"]["ext"]
                            )
                        ]
                    )
                    n_radar_frame = len(paths)
                    assert n_frame == n_radar_frame
                    radar_paths_chirp.append(paths)
                radar_paths = []
                for frame_id in range(n_frame):
                    frame_paths = []
                    for chirp_id in range(n_chirp):
                        frame_paths.append(radar_paths_chirp[chirp_id][frame_id])
                    radar_paths.append(frame_paths)
            elif radar_configs["data_type"] == "ROD2021":
                n_frame = len(os.listdir(radar_dir)) // len(radar_configs["chirp_ids"])
                # assert len(os.listdir(radar_dir)) == n_frame * len(
                #     radar_configs["chirp_ids"]
                # )
                radar_paths = []
                for frame_id in range(n_frame):
                    chirp_paths = []
                    for chirp_id in radar_configs["chirp_ids"]:
                        path = os.path.join(
                            radar_dir,
                            "%06d_%04d." % (frame_id, chirp_id)
                            + dataset.sensor_cfg.radar_cfg["ext"],
                        )
                        chirp_paths.append(path)
                    radar_paths.append(chirp_paths)
            else:
                raise ValueError

            data_dict = dict(
                data_root=data_root,
                data_path=seq_path,
                seq_name=seq,
                n_frame=n_frame,
                image_paths=image_paths,
                radar_paths=radar_paths,
                anno=None,
            )

            if split == "demo" or not os.path.exists(seq_anno_path):
                print("skip labels")
                # no labels need to be saved
                pickle.dump(data_dict, open(save_path, "wb"))
                continue
            else:
                anno_obj = {}
                if config_dict["dataset_cfg"]["anno_ext"] == ".txt":
                    anno_obj["metadata"] = load_anno_txt(
                        seq_anno_path, n_frame, dataset
                    )

                elif config_dict["dataset_cfg"]["anno_ext"] == ".json":
                    with open(os.path.join(seq_anno_path), "r") as f:
                        anno = json.load(f)
                    anno_obj["metadata"] = anno["metadata"]
                else:
                    raise

                anno_obj["confmaps"] = generate_confmaps(
                    anno_obj["metadata"], n_class, viz
                )
                data_dict["anno"] = anno_obj
                # save pkl files
                pickle.dump(data_dict, open(save_path, "wb"))
            # end frames loop

        except Exception as e:
            print("Error while preparing %s: %s" % (seq_path, e))


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    splits = args.split.split(",")
    out_data_dir = args.out_data_dir
    overwrite = args.overwrite

    dataset = CRUW(data_root=data_root, sensor_config_name="sensor_config_rod2021")
    config_dict = load_configs_from_file(args.config)
    radar_configs = dataset.sensor_cfg.radar_cfg

    for split in splits:
        if split not in SPLITS_LIST:
            raise TypeError("split %s cannot be recognized" % split)

    for split in splits:
        save_dir = os.path.join(out_data_dir, split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Preparing %s sets ..." % split)
        prepare_data(
            dataset,
            config_dict,
            out_data_dir,
            split,
            save_dir,
            viz=False,
            overwrite=overwrite,
        )

