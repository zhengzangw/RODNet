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
    type="CDC",
    name="ground_cdc_adam_lr4",
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
    use_noise_channel=True,
    # use_fft_channel=True,
)

train_cfg = dict(
    batch_size=3*16,
    n_epoch=30,
    lr=1e-4,
    lr_step=10, # 1e-4 -> 1e-14
    log_step=20,
    cropmix=True,
    # others
    win_size=16,
    train_step=1,
    train_stride=4,
    save_step=10000,
)
