import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "../cycleGAN/data/train"
VAL_DIR = "../cycleGAN/data/val"
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_WORKERS = 4
CURRENT_EPOCH = 0
NUM_EPOCHS = 300
LOAD_MODEL = True
SAVE_MODEL = True
ONLY_APPLY = False
CHECKPOINT = "classifier_checkpoint.pth.tar"
SIZE = 512

transforms = None
transforms_val_dataset = None

def def_transforms():
    global transforms
    global transforms_val_dataset
    transforms = A.Compose(
        [
            A.Resize(width=SIZE, height=SIZE),
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
            A.Resize(width=SIZE, height=SIZE),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
         ],
        additional_targets={"image0": "image"},
    )
