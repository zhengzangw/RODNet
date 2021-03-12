dataset_cfg = dict(
    dataset_name="rod2021v",
    base_root="./rod2021v",
    data_root="./rod2021v/sequences",
    anno_root="./rod2021v/annotations",
    anno_ext=".txt",
    train=dict(subdir="train",),
    test=dict(subdir="test",),
)

model_cfg = dict(
    type="CDC",
    name="valid_cdc_lr4_b4x4",
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
)

train_cfg = dict(
    n_epoch=10,
    batch_size=16, # 4x4
    lr=0.0001,
    lr_step=10,
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