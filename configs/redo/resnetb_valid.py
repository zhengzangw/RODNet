dataset_cfg = dict(
    dataset_name="rod2021_valid",
    base_root="./rod2021_valid",
    data_root="./rod2021_valid/sequences",
    anno_root="./rod2021_valid/annotations",
    anno_ext=".txt",
    train=dict(subdir="train",),
    test=dict(subdir="test",),
)

model_cfg = dict(
    name="resnetb",
    type="Resnet18b",
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
    # args
    use_noise_channel=True,
    double_hard=True,
)

train_cfg = dict(
    batch_size=2 * 8,
    n_epoch=50,
    lr=1e-5,
    lr_step=10,
    aug=True,
    log_step=20,
    # others
    win_size=16,
    train_step=1,
    train_stride=4,
    save_step=10000,
)
