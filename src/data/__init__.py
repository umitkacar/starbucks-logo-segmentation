import json
from multiprocessing import cpu_count
from typing import Optional, Sequence, Union

import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from mobile_seg.dataset import MaskDataset, load_df
from mylib.albumentations.augmentations.transforms import MyCoarseDropout


# noinspection PyAbstractClass
class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.params = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):

        train_list = [x for x in list(self.params[self.params["CLASS_DIR"]]["train"].values())]
        valid_list = [x for x in list(self.params[self.params["CLASS_DIR"]]["valid"].values())]

        # [CSV_PATH, IMG_PATH, MASK_PATH]
        df_train = load_df(train_list)
        # df_train = df_train.sample(n=100, random_state=1,replace=False)
        df_val = load_df(valid_list)

        class_dir = self.params["CLASS_DIR"]
        color_rgb = [self.params["CLASS_COLOR_RGB"][class_dir]]

        self.train_dataset = MaskDataset(
            df_train,
            class_dir,
            color_rgb,
            transform=A.Compose(
                [
                    A.Resize(
                        self.params["img_size"],
                        self.params["img_size"],
                    ),
                    # A.RandomResizedCrop(
                    #     self.params["img_size"],
                    #     self.params["img_size"],
                    # ),
                    A.Rotate(90, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.HueSaturationValue(p=0.3),
                    A.RGBShift(p=0.1),
                    A.RandomGamma(p=0.1),
                    A.IAASharpen(p=0.1),
                    A.OneOf(
                        [
                            A.ElasticTransform(p=0.3),
                            MyCoarseDropout(
                                p=0.5,
                                min_holes=1,
                                max_holes=8,
                                max_height=32,
                                max_width=32,
                            ),
                        ],
                        p=0.1,
                    ),
                    A.OneOf(
                        [
                            A.MultiplicativeNoise(p=0.5, per_channel=True, elementwise=True),
                            A.IAAAdditiveGaussianNoise(p=0.5),
                            A.IAAPerspective(p=0.5),
                        ],
                        p=0.1,
                    ),
                    A.OneOf(
                        [
                            A.OpticalDistortion(p=0.5),
                            A.GridDistortion(p=0.5),
                            A.IAAPiecewiseAffine(p=0.5),
                        ],
                        p=0.1,
                    ),
                ],
            ),
        )
        self.val_dataset = MaskDataset(
            df_val,
            class_dir,
            color_rgb,
            transform=A.Compose(
                [
                    A.Resize(
                        self.params["img_size"],
                        self.params["img_size"],
                    ),
                ],
            ),
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
        )
