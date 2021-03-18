import argparse
import json
import math
import os
import time

import rodnet.models
import torch
import torch.nn as nn
import torch.optim as optim
from cruw import CRUW
from rodnet.core.radar_processing import chirp_amp
from rodnet.datasets.collate_functions import cr_collate
from rodnet.datasets.CRDataLoader import CRDataLoader
from rodnet.datasets.CRDataset import CRDataset
from rodnet.datasets.CRDatasetSM import CRDatasetSM
from rodnet.utils.load_configs import load_configs_from_file
from rodnet.utils.solve_dir import create_dir_for_new_model
from rodnet.utils.visualization import visualize_train_img
from rodnet.warmup_scheduler.scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

NUM_WORKERS = 4


def set_args(name, args, model_cfg):
    val = None
    if name in model_cfg:
        val = model_cfg[name]
    if getattr(args, name) is True:
        val = True
    return val


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train RODNet.")
    parser.add_argument("--config", type=str, help="configuration file path")
    parser.add_argument(
        "--data_dir", type=str, default="./data/", help="directory to the prepared data"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./checkpoints/",
        help="directory to save trained model",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="path to the trained model"
    )
    parser.add_argument("--resume_opt", action="store_true")
    parser.add_argument(
        "--save_memory",
        action="store_true",
        help="use customized dataloader to save memory",
    )
    parser.add_argument("--use_noise_channel")
    parser.add_argument("--use_freq_channel", action="store_true")
    parser.add_argument("--use_fft_channel", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--seq_type", type=str, default=None)
    parser.add_argument("--double_hard", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # prepare
    args = parse_args()
    train_model_path = args.log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    is_cls = False

    # load config
    config_dict = load_configs_from_file(args.config)
    model_cfg = config_dict["model_cfg"]
    train_cfg = config_dict["train_cfg"]
    RODNet = rodnet.models.get_model(model_cfg["type"])
    n_epoch = train_cfg["n_epoch"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    aug_crop = True if "aug" in train_cfg else False

    # load dataset
    dataset = CRUW(
        data_root=config_dict["dataset_cfg"]["base_root"],
        sensor_config_name="sensor_config_rod2021",
    )
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid
    n_class_train = n_class = dataset.object_cfg.n_class
    channel = 2

    # Training skills
    use_noise_channel = set_args("use_noise_channel", args, model_cfg)
    use_freq_channel = set_args("use_freq_channel", args, model_cfg)
    use_fft_channel = set_args("use_fft_channel", args, model_cfg)
    double_hard = set_args("double_hard", args, model_cfg)

    if use_noise_channel:
        n_class_train += 1
        print("Use noise channel")
    if use_freq_channel:
        n_class_train += 2
        print("Use freq channel")
    if use_fft_channel:
        channel += 2
        print("Use fft channel")
    if "optim" in config_dict:
        opt = config_dict["optim"]
    else:
        opt = "Adam"
    print(f"Use optimizer: {opt}")
    if "stacked_num" in model_cfg:
        stacked_num = model_cfg["stacked_num"]
    else:
        stacked_num = None

    # create / load models
    cp_path = None
    epoch_start = 0
    iter_start = 0
    if args.resume_from is not None:
        assert os.path.exists(args.resume_from)
        cp_path = args.resume_from
        model_dir, model_name = create_dir_for_new_model(
            model_cfg["name"], train_model_path
        )
    else:
        model_dir, model_name = create_dir_for_new_model(
            model_cfg["name"], train_model_path
        )

    # logger
    writer = SummaryWriter(model_dir)
    save_config_dict = {
        "args": vars(args),
        "config_dict": config_dict,
    }
    config_json_name = os.path.join(
        model_dir, "config-" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    )
    with open(config_json_name, "w") as fp:
        json.dump(save_config_dict, fp)
    train_log_name = os.path.join(model_dir, "train.log")
    with open(train_log_name, "w"):
        pass

    print(
        "Building dataloader ... (Mode: %s)"
        % ("save_memory" if args.save_memory else "normal")
    )

    if aug_crop:
        print("Use video mix augmentation.")
    crdata_train = CRDataset(
        data_dir=args.data_dir,
        dataset=dataset,
        config_dict=config_dict,
        split="train",
        noise_channel=use_noise_channel,
        freq_channel=use_freq_channel,
        fft_channel=use_fft_channel,
        seq_type=args.seq_type,
        aug_crop=aug_crop,
        double_hard=double_hard,
    )
    seq_names = crdata_train.seq_names
    index_mapping = crdata_train.index_mapping
    dataloader = DataLoader(
        crdata_train,
        batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=cr_collate,
    )

    print("Building model ... (%s)" % model_cfg)
    print("Training config ... (%s)" % train_cfg)
    if model_cfg["type"].startswith("cls"):
        rodnet = RODNet(model_cfg["n_class"])
        criterion = nn.CrossEntropyLoss()
        is_cls = True
    elif model_cfg["type"] in ["CDC", "C21D", "CDCD", "GSC", "GSCmp", "Resnet18"]:
        rodnet = RODNet(n_class_train, n_channel=channel)
        criterion = nn.MSELoss()
    elif model_cfg["type"] in ["HG", "HGwI"]:
        rodnet = RODNet(n_class_train, stacked_num=stacked_num)
        criterion = nn.BCELoss()
    else:
        raise TypeError
    if args.parallel:
        rodnet = nn.DataParallel(rodnet).cuda()
    else:
        rodnet = rodnet.cuda()
    criterion = criterion.cuda()

    if opt == "Adam":
        optimizer = optim.Adam(rodnet.parameters(), lr=lr)
    elif opt == "SGD":
        optimizer = optim.SGD(
            rodnet.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5
        )
    else:
        raise NotImplementedError

    if "lr_step" in train_cfg:
        scheduler = StepLR(optimizer, step_size=train_cfg["lr_step"], gamma=0.1)
    elif "restart_epoch" in train_cfg:
        scheduler = CosineAnnealingWarmRestarts(optimizer, train_cfg["restart_epoch"])
    elif "warmup_cosine" in train_cfg:
        print("Use warmup cosine scheduler.")
        t = 10
        T = 50
        lambda1 = lambda epoch: (
            (0.9 * epoch / t + 0.1)
            if epoch < t
            else 0.5 * (1 + math.cos(math.pi * (epoch - t) / (T - t)))
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise NotImplementedError

    iter_count = 0
    if cp_path is not None:
        print(f"Load from {cp_path}")
        checkpoint = torch.load(cp_path)
        if "optimizer_state_dict" in checkpoint:
            rodnet.load_state_dict(checkpoint["model_state_dict"])
            if args.resume_opt:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epoch_start = checkpoint["epoch"] + 1
                iter_start = checkpoint["iter"] + 1
                loss_cp = checkpoint["loss"]
                if "iter_count" in checkpoint:
                    iter_count = checkpoint["iter_count"]
        else:
            rodnet.load_state_dict(checkpoint)

    # print training configurations
    print("Model name: %s" % model_name)
    print("Number of sequences to train: %d" % crdata_train.n_seq)
    print("Training dataset length: %d" % len(crdata_train))
    print("Batch size: %d" % batch_size)
    print(
        "Number of iterations in each epoch: %d" % int(len(crdata_train) / batch_size)
    )

    for epoch in range(epoch_start, n_epoch + 1):
        tic_load = time.time()

        for iter, data_dict in enumerate(dataloader):

            data = data_dict["radar_data"]
            image_paths = data_dict["image_paths"]
            confmap_gt = data_dict["anno"]["confmaps"]
            seq_type = data_dict["seq_type"]
            tic = time.time()
            optimizer.zero_grad()  # zero the parameter gradients
            
            confmap_preds = rodnet(data.float().cuda())

            loss_confmap = 0
            if is_cls:
                loss_confmap = criterion(confmap_preds, seq_type.cuda())
            elif stacked_num is not None:
                for i in range(stacked_num):
                    loss_cur = criterion(confmap_preds[i], confmap_gt.float().cuda())
                    loss_confmap += loss_cur
            else:
                loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())

            loss_confmap.backward()
            optimizer.step()

            if iter % train_cfg["log_step"] == 0:
                # print statistics
                stats = (
                    "epoch %2d, iter %4d: loss: %.8f | lr: %.8f | load time: %.4f | backward time: %.4f"
                    % (
                        epoch + 1,
                        iter + 1,
                        loss_confmap.item(),
                        get_lr(optimizer),
                        tic - tic_load,
                        time.time() - tic,
                    )
                )
                print(stats)
                with open(train_log_name, "a+") as f_log:
                    f_log.write(stats + "\n")

                if stacked_num is not None:
                    writer.add_scalar("loss/loss_all", loss_confmap.item(), iter_count)
                    # confmap_pred = confmap_preds[stacked_num - 1].cpu().detach().numpy()
                else:
                    writer.add_scalar("loss/loss_all", loss_confmap.item(), iter_count)
                    # confmap_pred = confmap_preds.cpu().detach().numpy()

                # draw train images
                # fig_name = os.path.join(
                #     train_viz_path,
                #     "%03d_%010d_%06d.png" % (epoch + 1, iter_count, iter + 1),
                # )
                # img_path = image_paths[0][0]
                # visualize_train_img(
                #     fig_name,
                #     img_path,
                #     chirp_amp_curr,
                #     confmap_pred[0, :n_class, 0, :, :],
                #     confmap_gt[0, :n_class, 0, :, :],
                # )

            if (iter + 1) % train_cfg["save_step"] == 0:
                # validate current model
                # print("validing current model ...")
                # validate()

                # save current model
                print("saving current model ...")
                status_dict = {
                    "model_name": model_name,
                    "epoch": epoch,
                    "iter": iter,
                    "model_state_dict": rodnet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_confmap,
                    "iter_count": iter_count,
                }
                save_model_path = "%s/epoch_%02d_iter_%010d.pkl" % (
                    model_dir,
                    epoch + 1,
                    iter_count + 1,
                )
                torch.save(status_dict, save_model_path)

            iter_count += 1
            tic_load = time.time()

        # save current model
        print("saving current epoch model ...")
        status_dict = {
            "model_name": model_name,
            "epoch": epoch,
            "iter": iter,
            "model_state_dict": rodnet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_confmap,
            "iter_count": iter_count,
        }
        save_model_path = "%s/epoch_%02d_final.pkl" % (model_dir, epoch + 1)
        torch.save(status_dict, save_model_path)

        scheduler.step()

    print("Training Finished.")

