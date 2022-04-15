import random
import torch
import torch.nn as nn
import os
import numpy as np
import copy

import config

def save_checkpoint(model, optimizer, epoch: int, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, change_current_epoch=False):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if change_current_epoch:
        config.CURRENT_EPOCH = checkpoint["epoch"]


def seed_everything(seed=42):
    # not used rn
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directory(directory_path: str):
    """
    :param directory_path: path of the directory to be created
    :return: nothing
    """
    try:
        os.mkdir(directory_path)
    except FileExistsError:
        pass
