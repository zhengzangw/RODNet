dataset_cfg = dict(
    dataset_name="rod2021",
    base_root="./rod2021",
    data_root="./rod2021/sequences",
    anno_root="./rod2021/annotations",
    anno_ext=".txt",
    train=dict(subdir="train",),
    valid=dict(subdir="valid", seqs=[],),
    test=dict(subdir="test",),
    demo=dict(subdir="demo", seqs=[],),
)

model_cfg = dict(
    type="HG",
    name="rodnet-hg1-win16-wobg",
    max_dets=20,
    peak_thres=0.33,
    ols_thres=0.3,
    stacked_num=1,
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
