dataset_cfg = dict(
    dataset_name="ROD2021",
    base_root="./ROD2021",
    data_root="./ROD2021/sequences",
    anno_root="./ROD2021/annotations",
    anno_ext=".txt",
    train=dict(
        subdir="train",
        # seqs=[],  # can choose from the subdir folder
        seqs=[
            "2019_04_09_BMS1000",
            "2019_04_09_BMS1001",
            "2019_04_09_BMS1002",
            "2019_04_09_CMS1002",
            # "2019_04_09_PMS1000",
            "2019_04_09_PMS1001",
            "2019_04_09_PMS2000",
            "2019_04_09_PMS3000",
            "2019_04_30_MLMS000",
            # "2019_04_30_MLMS001",
            "2019_04_30_MLMS002",
            "2019_04_30_PBMS002",
            "2019_04_30_PBMS003",
            "2019_04_30_PCMS001",
            # "2019_04_30_PM2S003",
            "2019_04_30_PM2S004",
            "2019_05_09_BM1S008",
            "2019_05_09_CM1S004",
            "2019_05_09_MLMS003",
            # "2019_05_09_PBMS004",
            "2019_05_09_PCMS002",
            "2019_05_23_PM1S012",
            "2019_05_23_PM1S013",
            "2019_05_23_PM1S014",
            # "2019_05_23_PM1S015",
            "2019_05_23_PM2S011",
            "2019_05_29_BCMS000",
            "2019_05_29_BM1S016",
            "2019_05_29_BM1S017",
            # "2019_05_29_MLMS006",
            "2019_05_29_PBMS007",
            "2019_05_29_PCMS005",
            "2019_05_29_PM2S015",
            "2019_05_29_PM3S000",
            # "2019_09_29_ONRD001",
            "2019_09_29_ONRD002",
            "2019_09_29_ONRD005",
            "2019_09_29_ONRD006",
            "2019_09_29_ONRD011",
            # "2019_09_29_ONRD013",
        ],
    ),
    test=dict(
        subdir="train",
        seqs=[
            "2019_04_09_PMS1000",
            "2019_04_30_MLMS001",
            "2019_04_30_PM2S003",
            "2019_05_09_PBMS004",
            "2019_05_23_PM1S015",
            "2019_05_29_MLMS006",
            "2019_09_29_ONRD001",
            "2019_09_29_ONRD013",
        ],
    ),
    # test=dict(
    #     subdir="test",
    #     # seqs=[],  # can choose from the subdir folder
    # ),
    # demo=dict(subdir="demo", seqs=[],),
)

model_cfg = dict(
    type="CDC",
    name="rodnet-cdc-win16-wobg",
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
)

confmap_cfg = dict(
    confmap_sigmas={
        "pedestrian": 15,
        "cyclist": 20,
        "car": 30,
        # 'van': 40,
        # 'truck': 50,
    },
    confmap_sigmas_interval={
        "pedestrian": [5, 15],
        "cyclist": [8, 20],
        "car": [10, 30],
        # 'van': [15, 40],
        # 'truck': [20, 50],
    },
    confmap_length={
        "pedestrian": 1,
        "cyclist": 2,
        "car": 3,
        # 'van': 4,
        # 'truck': 5,
    },
)

train_cfg = dict(
    n_epoch=100,
    batch_size=4,
    lr=0.00001,
    lr_step=5,  # lr will decrease 10 times after lr_step epoches
    win_size=16,
    train_step=1,
    train_stride=4,
    log_step=100,
    save_step=10000,
)
test_cfg = dict(
    test_step=1,
    test_stride=8,
    rr_min=1.0,  # min radar range
    rr_max=20.0,  # max radar range
    ra_min=-60.0,  # min radar angle
    ra_max=60.0,  # max radar angle
)

# python tools/prepare_dataset/prepare_data.py         --config configs/config_rodnet_cdc_win16.py --data_root ROD2021 --split train,valid --out_data_dir data/rod2021_cdc_val
