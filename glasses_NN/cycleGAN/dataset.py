import os
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset


class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        if index >= self.zebra_len:
            index_randomise = random.randint(0, self.zebra_len - 1)
            zebra_img = self.zebra_images[index_randomise]
        else:
            zebra_img = self.zebra_images[index]

        if index >= self.horse_len:
            index_randomise = random.randint(0, self.horse_len - 1)
            horse_img = self.horse_images[index_randomise]
        else:
            horse_img = self.horse_images[index]

        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]
            # see the transform definition in config.py and https://albumentations.ai/docs/examples/example_multi_target
            # for more details

        return zebra_img, horse_img
