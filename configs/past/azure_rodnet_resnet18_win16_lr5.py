dataset_cfg = dict(
    dataset_name="rod2021_valid",
    base_root="./rod2021_valid",
    data_root="./rod2021_valid/sequences",
    anno_root="./rod2021_valid/annotations",
    anno_ext=".txt",
    train=dict(subdir="train",),
    valid=dict(subdir="valid", seqs=[],),
    test=dict(subdir="test",),
    demo=dict(subdir="demo", seqs=[],),
)

model_cfg = dict(
    type="Resnet18",
    name="rodnet-resnet18-win16",
    max_dets=20,
    peak_thres=0.4,
    ols_thres=0.3,
)

confmap_cfg = dict(
    confmap_sigmas={"pedestrian": 15, "cyclist": 20, "car": 30,},
    confmap_sigmas_interval={
        "pedestrian": [5, 15],
        "cyclist": [8, 20],
        "car": [10, 30],
    },
    confmap_length={"pedestrian": 1, "cyclist": 2, "car": 3,},
)

train_cfg = dict(
    n_epoch=30,
    batch_size=32,
    lr=0.0001,
    lr_step=10,  # lr will decrease 10 times after lr_step epoches
    win_size=16,
    train_step=1,
    train_stride=4,
    log_step=20,
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
