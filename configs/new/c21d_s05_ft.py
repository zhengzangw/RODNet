dataset_cfg = dict(
    dataset_name="rod2021",
    base_root="./rod2021",
    data_root="./rod2021/sequences",
    anno_root="./rod2021/annotations",
    anno_ext=".txt",
    train=dict(subdir="train",),
    test=dict(subdir="test",),
)

model_cfg = dict(
    name="c21d_s05",
    type="C21D",
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
    # args
    use_noise_channel=True,
    # double_hard=True,
)

train_cfg = dict(
    batch_size=4 * 2,
    n_epoch=30,
    lr=1e-5,
    # warmup_cosine=True,
    lr_step=10,
    aug=True,
    log_step=20,
    # others
    win_size=16,
    train_step=1,
    train_stride=4,
    save_step=10000,
)
