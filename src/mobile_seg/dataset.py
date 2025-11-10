import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

pd.set_option("display.max_colwidth", None)
pd.options.display.float_format = "{:,.4f}".format
pd.options.display.max_columns = 500

from pathlib import Path

from mylib.pandas.cache import pd_cache

CACHE_DIR = Path("../cache")


@pd_cache(CACHE_DIR, ext=".pqt")
def load_df(path_list):

    # [CSV_PATH, IMG_PATH, MASK_PATH]
    mask_paths = []
    img_paths = []
    for path in path_list:

        pd_data = pd.read_csv(path[0], header=None, names=["img_path"])
        pd_data = pd_data["img_path"].tolist()

        for f_name in pd_data:
            img_paths.append(f'{path[1]}/{f_name.rsplit(".")[0]}.jpg')
            mask_paths.append(f'{path[2]}/{f_name.rsplit(".")[0]}.png')

    # img_paths = img_paths[:4]
    # mask_paths = mask_paths[:4]

    return pd.DataFrame(
        {
            "img_path": map(str, img_paths),
            "mask_path": map(str, mask_paths),
        },
    )


class MaskDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_dir,
        color_rgb,
        transform: A.Compose,
    ):
        self.df = df
        self.transform = transform
        self.class_dir = class_dir
        self.color_rgb = color_rgb

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # print(row['img_path'])
        img = cv2.imread(row["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)

        mask = cv2.imread(row["mask_path"])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask_gt = [
            (mask[:, :, 0] == v[0]) & (mask[:, :, 1] == v[1]) & (mask[:, :, 2] == v[2])
            for v in self.color_rgb
        ]
        mask_gt = np.array(mask_gt).squeeze(0).astype("float32")

        augmented = self.transform(image=img, mask=mask_gt)

        img = np.array(augmented["image"]).astype(np.float32).transpose((2, 0, 1))
        mask_gt = np.array(augmented["mask"]).astype(np.float32)

        if self.class_dir == "multi_ten":
            mask_gt = np.array(np.argmax(mask_gt, axis=-1)).astype(np.float32)
            mask_gt = torch.from_numpy(mask_gt).long()
            img = torch.from_numpy(img)

        return img / 255.0, mask_gt

    def __len__(self):
        return len(self.df)


class MaskDatasetValid(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_dir,
        color_rgb,
        transform: A.Compose,
    ):
        self.df = df
        self.transform = transform
        self.class_dir = class_dir
        self.color_rgb = color_rgb

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        print(row["img_path"])
        img = cv2.imread(row["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)

        mask = cv2.imread(row["mask_path"])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask_gt = [
            (mask[:, :, 0] == v[0]) & (mask[:, :, 1] == v[1]) & (mask[:, :, 2] == v[2])
            for v in self.color_rgb
        ]
        mask_gt = np.array(mask_gt).squeeze(0).astype("float32")

        augmented = self.transform(image=img, mask=mask_gt)

        img = np.array(augmented["image"]).astype(np.float32).transpose((2, 0, 1))
        mask_gt = np.array(augmented["mask"]).astype(np.float32)

        if self.class_dir == "multi_ten":
            mask_gt = np.array(np.argmax(mask_gt, axis=-1)).astype(np.float32)
            mask_gt = torch.from_numpy(mask_gt)
            img = torch.from_numpy(img)

        return img / 255.0, mask_gt, row["img_path"]

    def __len__(self):
        return len(self.df)
