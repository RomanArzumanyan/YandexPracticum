import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from transformers import AutoTokenizer

import albumentations as A
import matplotlib.pyplot as plt


def plot_hist(arr: np.ndarray, xlabel: str):
    plt.ylabel('num dishes')
    plt.xlabel(xlabel)
    plt.hist(arr, bins=30, color='skyblue', edgecolor='black')
    plt.show(block=False)


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms=None, ds_type="train"):
        df = pd.read_csv(config.CSV_PATH)
        self.df = df[df['split'] == ds_type].reset_index()

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms
        self.ds_type = ds_type
        self.img_path = config.IMG_PATH

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dish_id = self.df.loc[idx, "dish_id"]
        image = Image.open(os.path.join(
            self.img_path, dish_id, 'rgb.png')).convert('RGB')
        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]
        mass = self.df.loc[idx, "total_mass"]
        calories = self.df.loc[idx, "total_calories"]
        ingredients = ' '.join(self.df.loc[idx, "ingredients"].split(';'))

        return {
            "ingredients": ingredients,
            "calories": calories / mass,
            "dish_id": dish_id,
            "image": image,
            "mass": mass,
        }

    def summary(self):
        '''
        Output dataset summary 
        '''
        num_ingredients = []
        calories = []
        calories_norm = []

        for i in tqdm(range(0, len(self))):
            entry = self[i]
            num_ingredients.append(len(entry['ingredients'].split(' ')))
            calories.append(entry['calories'] * entry['mass'])
            calories_norm.append(entry['calories'])

        return {
            "ingridients num": np.asarray(num_ingredients, dtype=np.int32),
            "calories / g": np.asarray(calories_norm, dtype=np.float32),
            "calories": np.asarray(calories, dtype=np.int32),
        }


def collate_fn(batch, tokenizer):
    ingredients = [item["ingredients"] for item in batch]
    calories = torch.FloatTensor([item["calories"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    mass = torch.FloatTensor([item["mass"] for item in batch])
    id = [item["dish_id"] for item in batch]

    tokenized_input = tokenizer(ingredients,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "id": id,
        "mass": mass,
        "label": calories,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.RandomCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.1,
                              p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )

    return transforms
