import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 6  # will be overwritten if config.SIZE > 256 (cf. train.py)
LEARNING_RATE = 1e-5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
CURRENT_EPOCH = 0
NUM_EPOCHS = 300
LOAD_MODEL = True
SAVE_MODEL = True
ONLY_GENERATE = False
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

LAMBDA_IDENTITY = "argparse"
ONE_SIDED_LABEL_SMOOTHING = "argparse"
HORSES_CLASS = "argparse"
ZEBRAS_CLASS = "argparse"
SIZE = "argparse"
SKIP_CONNECTION = "argparse"

transforms = None
transforms_val_dataset = None

def def_transforms():
    global transforms
    global transforms_val_dataset
    transforms = A.Compose(
        [
            A.Resize(width=SIZE, height=SIZE),  # noqa
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
         ],
        additional_targets={"image0": "image"},
        # Note about additional_targets:
        # Sometimes you want to apply the same set of augmentations to multiple input objects of the same type.
        # For example, you might have a set of frames from the video, and you want to augment them in the same way.
    )

    transforms_val_dataset = A.Compose(
        [
            A.Resize(width=SIZE, height=SIZE),  # noqa
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
         ],
        additional_targets={"image0": "image"},
    )
