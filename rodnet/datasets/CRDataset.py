import os
import pickle
import random
import time

import numpy as np
from torch.utils import data
from tqdm import tqdm

from .loaders import list_pkl_filenames, list_pkl_filenames_from_prepared


def add_freq_channel(confmap, radar_data):
    # confmap [n_class, W, h, w]
    # radar_data [2, W, h, w]

    energy = np.expand_dims(np.sqrt(radar_data[0] ** 2 + radar_data[1] ** 2), axis=0)
    phase = np.expand_dims(np.arctan(radar_data[1] / radar_data[0]), axis=0)
    confmap = np.concatenate([confmap, energy, phase])
    return confmap


def add_fft_channel(radar_data):
    radar_data_c = radar_data[:, :, :, 0] + radar_data[:, :, :, 1] * 1j
    radar_data_fft = np.fft.fft(radar_data_c, axis=0)
    radar_data_fft_r = np.real(radar_data_fft)
    radar_data_fft_c = np.imag(radar_data_fft)
    radar_data_fft = np.stack([radar_data_fft_r, radar_data_fft_c], axis=-1)
    radar_data = np.concatenate([radar_data, radar_data_fft], axis=-1)
    return radar_data


def flipcoin(p):
    return random.random() < p


def cropmix(dict_1, dict_2):
    radar1 = dict_1["radar_data"]
    radar2 = dict_2["radar_data"]
    confmap1 = dict_1["anno"]["confmaps"]
    confmap2 = dict_2["anno"]["confmaps"]

    W_x = random.randint(32, 96) #128
    W_y = random.randint(32, 96) #128
    x_l = random.randint(0, 127 - W_x)
    y_l = random.randint(0, 127 - W_y)

    radar1[:, :, x_l : x_l + W_x, y_l : y_l + W_y] = radar2[
        :, :, x_l : x_l + W_x, y_l : y_l + W_y
    ]
    confmap1[:, :, x_l : x_l + W_x, y_l : y_l + W_y] = confmap2[
        :, :, x_l : x_l + W_x, y_l : y_l + W_y
    ]

    dict_1["radar_data"] = radar1
    dict_1["anno"]["confmaps"] = confmap1
    return dict_1


def mix(dict_1, dict_2):
    radar1 = dict_1["radar_data"]
    radar2 = dict_2["radar_data"]
    confmap1 = dict_1["anno"]["confmaps"]
    confmap2 = dict_2["anno"]["confmaps"]

    alpha = random.random()
    radar1 = radar1 * alpha + radar2 * (1 - alpha)
    confmap1 = confmap1 * alpha + confmap2 * (1 - alpha)

    dict_1["radar_data"] = radar1
    dict_1["anno"]["confmaps"] = confmap1
    return dict_1


def noise(dict_1, dict_2):
    radar1 = dict_1["radar_data"]
    radar2 = dict_2["radar_data"]
    confmap1 = dict_1["anno"]["confmaps"]
    confmap2 = dict_2["anno"]["confmaps"]

    channal = radar1.shape[0]
    assert confmap1.shape[0] == 4
    info_mask = confmap2[-1] < 0.99999999
    info_mask = np.stack([info_mask] * channal)
    radar2[info_mask] = 0
    radar1 += radar2

    dict_1["radar_data"] = radar1
    dict_1["anno"]["confmaps"] = confmap1
    return dict_1


# seq, train, dynamic, scene
# dynamic 1
# parking lot (PL), campus road (CR), city street (CS), and highway (HW)
scene_id = {"PL": 0, "CR": 1, "CS": 2, "HW": 3}
seq_info = {
    "2019_04_09_BMS1000": (1, 0, "PL"),
    "2019_04_09_BMS1001": (1, 0, "PL"),
    "2019_04_09_BMS1002": (1, 0, "PL"),
    "2019_04_09_PMS1000": (1, 0, "PL"),
    "2019_04_09_PMS1001": (1, 0, "PL"),
    "2019_04_09_PMS2000": (1, 0, "PL"),
    "2019_04_09_PMS3000": (1, 0, "PL"),
    "2019_04_09_CMS1002": (1, 0, "PL"),
    "2019_04_30_PBMS003": (1, 0, "PL"),
    "2019_04_30_PM2S003": (1, 0, "PL"),
    "2019_05_23_PM2S011": (1, 0, "PL"),
    "2019_05_23_PM1S012": (1, 0, "PL"),
    "2019_05_23_PM1S015": (1, 0, "PL"),
    "2019_05_23_PM1S013": (1, 0, "PL"),
    "2019_05_23_PM1S014": (1, 0, "PL"),
    "2019_05_29_BM1S016": (1, 0, "PL"),
    "2019_05_29_BM1S017": (1, 0, "PL"),
    "2019_05_29_PBMS007": (1, 0, "PL"),
    "2019_05_29_PM2S015": (1, 0, "PL"),
    "2019_05_29_PM3S000": (1, 0, "PL"),
    "2019_04_30_PBMS002": (1, 0, "PL"),
    "2019_04_30_PM2S004": (1, 0, "PL"),
    "2019_04_30_MLMS000": (1, 0, "CR"),
    "2019_05_09_BM1S008": (1, 0, "CR"),
    "2019_05_09_PCMS002": (1, 0, "CR"),
    "2019_04_30_MLMS001": (1, 0, "CR"),
    "2019_04_30_PCMS001": (1, 0, "CR"),
    "2019_05_09_CM1S004": (1, 0, "CR"),
    "2019_05_29_MLMS006": (1, 0, "CR"),
    "2019_04_30_MLMS002": (1, 0, "CR"),
    "2019_05_09_MLMS003": (1, 0, "CR"),
    "2019_05_29_BCMS000": (1, 0, "CR"),
    "2019_05_09_PBMS004": (1, 0, "CR"),
    "2019_05_29_PCMS005": (1, 0, "CR"),
    "2019_09_29_ONRD002": (1, 1, "CS"),
    "2019_09_29_ONRD013": (1, 1, "CS"),
    "2019_09_29_ONRD001": (1, 1, "CS"),
    "2019_09_29_ONRD011": (1, 1, "HW"),
    "2019_09_29_ONRD006": (1, 1, "HW"),
    "2019_09_29_ONRD005": (1, 1, "HW"),
    # test
    "2019_05_28_CM1S013": (0, 0, "CR"),
    "2019_05_28_MLMS005": (0, 0, "PL"),
    "2019_05_28_PBMS006": (0, 0, "PL"),
    "2019_05_28_PCMS004": (0, 0, "PL"),
    "2019_05_28_PM2S012": (0, 0, "PL"),
    "2019_05_28_PM2S014": (0, 0, "PL"),
    "2019_09_18_ONRD004": (0, 1, "HW"),
    "2019_09_18_ONRD009": (0, 1, "HW"),
    "2019_09_29_ONRD012": (0, 1, "HW"),
    "2019_10_13_ONRD048": (0, 1, "HW"),
}


def seq_split(data_files, seq_type):
    ret = []
    assert seq_type in ["PL", "CR", "CS", "HW", "D", "S"]
    for data_file in data_files:
        seq_name = data_file.split(".")[0]
        flag = (
            (seq_type == "D" and seq_info[seq_name][1] == 1)
            or (seq_type == "S" and seq_info[seq_name][1] == 0)
            or seq_info[seq_name][2] == seq_type
        )
        if flag:
            ret.append(data_file)

    return ret


def double_hard_seq(data_files):
    ret = []
    for data_file in data_files:
        seq_name = data_file.split(".")[0]
        if seq_info[seq_name][1] == 1:
            ret.append(data_file)
            ret.append(data_file)
        else:
            ret.append(data_file)
    return ret


class CRDataset(data.Dataset):
    """
    Pytorch Dataloader for CR Dataset
    :param detail_dir: data details directory
    :param confmap_dir: confidence maps directory
    :param win_size: seqence window size
    :param n_class: number of classes for detection
    :param step: frame step inside each sequence
    :param stride: data sampling
    :param set_type: train, valid, test
    :param is_random_chirp: random load chirp or not
    """

    def __init__(
        self,
        data_dir,
        dataset,
        config_dict,
        split,
        is_random_chirp=True,
        subset=None,
        noise_channel=False,
        freq_channel=False,
        fft_channel=False,
        seq_type=None,
        aug_crop=False,
        double_hard=False,
    ):
        # parameters settings
        self.aug_crop = aug_crop
        self.fft_channel = fft_channel
        self.data_dir = data_dir
        self.dataset = dataset
        self.config_dict = config_dict
        self.n_class = dataset.object_cfg.n_class
        self.win_size = config_dict["train_cfg"]["win_size"]
        self.split = split
        if split == "train" or split == "valid":
            self.step = config_dict["train_cfg"]["train_step"]
            self.stride = config_dict["train_cfg"]["train_stride"]
        else:
            self.step = config_dict["test_cfg"]["test_step"]
            self.stride = config_dict["test_cfg"]["test_stride"]
        self.is_random_chirp = is_random_chirp
        self.n_chirps = 1
        self.noise_channel = noise_channel
        self.freq_channel = freq_channel

        # dataset initialization
        self.image_paths = []
        self.radar_paths = []
        self.obj_infos = []
        self.confmaps = []
        self.n_data = 0
        self.index_mapping = []

        if subset is not None:
            self.data_files = [subset + ".pkl"]
        else:
            self.data_files = list_pkl_filenames_from_prepared(data_dir, split)
        if seq_type is not None:
            self.data_files = seq_split(self.data_files, seq_type)
        if double_hard:
            self.data_files = double_hard_seq(self.data_files)

        self.seq_names = [name.split(".")[0] for name in self.data_files]
        self.n_seq = len(self.seq_names)

        split_folder = split
        for seq_id, data_file in tqdm(enumerate(self.data_files), leave=False):
            data_file_path = os.path.join(data_dir, split_folder, data_file)
            data_details = pickle.load(open(data_file_path, "rb"))
            if split == "train" or split == "valid":
                assert data_details["anno"] is not None
            n_frame = data_details["n_frame"]
            self.image_paths.append(data_details["image_paths"])
            self.radar_paths.append(data_details["radar_paths"])
            n_data_in_seq = (
                n_frame - (self.win_size * self.step - 1)
            ) // self.stride + (
                1
                if (n_frame - (self.win_size * self.step - 1)) % self.stride > 0
                else 0
            )
            self.n_data += n_data_in_seq
            for data_id in range(n_data_in_seq):
                self.index_mapping.append([seq_id, data_id * self.stride])
            if data_details["anno"] is not None:
                self.obj_infos.append(data_details["anno"]["metadata"])
                self.confmaps.append(data_details["anno"]["confmaps"])

    def __len__(self):
        """Total number of data/label pairs"""
        return self.n_data

    def get_item(self, index):
        seq_id, data_id = self.index_mapping[index]
        seq_name = self.seq_names[seq_id]
        seq_type = None
        if seq_name in seq_info and seq_info[seq_name][-1] in scene_id:
            seq_type = scene_id[seq_info[seq_name][-1]]
        image_paths = self.image_paths[seq_id]
        radar_paths = self.radar_paths[seq_id]
        if len(self.confmaps) != 0:
            this_seq_obj_info = self.obj_infos[seq_id]
            this_seq_confmap = self.confmaps[seq_id]

        data_dict = dict(status=True, seq_names=seq_name, image_paths=[])
        data_dict["seq_type"] = seq_type

        if self.is_random_chirp:
            chirp_id = random.randint(
                0, self.dataset.sensor_cfg.radar_cfg["n_chirps"] - 1
            )
        else:
            chirp_id = 0

        radar_configs = self.dataset.sensor_cfg.radar_cfg
        ramap_rsize = radar_configs["ramap_rsize"]
        ramap_asize = radar_configs["ramap_asize"]

        # Load radar data
        try:
            if radar_configs["data_type"] == "ROD2021":
                radar_npy_win = np.zeros(
                    (self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32
                )
                chirp_id = 0  # only use chirp 0 for training
                for idx, frameid in enumerate(
                    range(data_id, data_id + self.win_size * self.step, self.step)
                ):
                    radar_npy_win[idx, :, :, :] = np.load(
                        radar_paths[frameid][chirp_id]
                    )
                    data_dict["image_paths"].append(image_paths[frameid])
            else:
                raise NotImplementedError
        except:
            # in case load npy fail
            data_dict["status"] = False
            if not os.path.exists("./tmp"):
                os.makedirs("./tmp")
            log_name = "loadnpyfail-" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
            with open(os.path.join("./tmp", log_name), "w") as f_log:
                f_log.write(
                    "npy path: "
                    + radar_paths[frameid][chirp_id]
                    + "\nframe indices: %d:%d:%d"
                    % (data_id, data_id + self.win_size * self.step, self.step)
                )
            return data_dict

        C = 2
        if self.fft_channel:
            C += 2
            radar_npy_win = add_fft_channel(radar_npy_win)
        radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))

        assert radar_npy_win.shape == (
            C,
            self.win_size,
            radar_configs["ramap_rsize"],
            radar_configs["ramap_asize"],
        )

        data_dict["radar_data"] = radar_npy_win
        data_dict["aug"] = dict(crop=0, mix=0, noise=0)

        # Load annotations
        if len(self.confmaps) != 0:
            confmap_gt = this_seq_confmap[
                data_id : data_id + self.win_size * self.step : self.step
            ]
            confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            obj_info = this_seq_obj_info[
                data_id : data_id + self.win_size * self.step : self.step
            ]
            if self.noise_channel:
                assert confmap_gt.shape == (
                    self.n_class + 1,
                    self.win_size,
                    radar_configs["ramap_rsize"],
                    radar_configs["ramap_asize"],
                )
            else:
                confmap_gt = confmap_gt[: self.n_class]
                assert confmap_gt.shape == (
                    self.n_class,
                    self.win_size,
                    radar_configs["ramap_rsize"],
                    radar_configs["ramap_asize"],
                )
            if self.freq_channel:
                confmap_gt = add_freq_channel(confmap_gt, radar_npy_win)

            data_dict["anno"] = dict(obj_infos=obj_info, confmaps=confmap_gt,)
        else:
            data_dict["anno"] = None
        return data_dict

    def rand_item(self):
        ind = random.randint(0, len(self.index_mapping) - 1)
        rand_dict = self.get_item(ind)
        return rand_dict

    def augmentation(self, data_dict):
        if flipcoin(1 / 2):
            data_dict = cropmix(data_dict, self.rand_item())
            data_dict["aug"]["crop"] = 1
        elif flipcoin(1 / 2):
            data_dict = mix(data_dict, self.rand_item())
            data_dict["aug"]["mix"] = 1
        if flipcoin(1 / 2):
            # noise
            data_dict = noise(data_dict, self.rand_item())
            data_dict["aug"]["noise"] = 1

        return data_dict

    def __getitem__(self, index):
        # return data_dict["seq_type", "radar_data", "anno", "aug"]
        data_dict = self.get_item(index)
        if self.aug_crop:
            data_dict = self.augmentation(data_dict)
        return data_dict
