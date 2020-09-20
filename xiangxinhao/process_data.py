import os

import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

import numpy as np


class SIIM_ISIC(torch.utils.data.Dataset):
    def __init__(self, data_root='/home/group3/DataSet', train=True, transform=None):

        self.data_root = data_root
        self.train = train
        self.transform = transform

        if train:
            self.df = pd.read_csv(os.path.join(data_root, 'training_set.csv'))
            self.imageFolder = os.path.join(data_root, 'Training_set')
        else:
            self.df = pd.read_csv(os.path.join(data_root, 'validation_set.csv'))
            self.imageFolder = os.path.join(data_root, 'Validation_set')

        self.df['sex'].fillna('unknown', inplace=True)
        self.df['age_approx'].fillna(-1, inplace=True)
        self.df['anatom_site_general_challenge'].fillna('unknown', inplace=True)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]["image_name"]
        path = os.path.join(self.imageFolder, '{}.jpg'.format(image_name))
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        target = self.df.iloc[idx]["target"]

        sex = self.df.iloc[idx]["sex"]
        age_approx = self.df.iloc[idx]["age_approx"]
        anatom_site_general_challenge = self.df.iloc[idx]["anatom_site_general_challenge"]
        meta = {
            "sex": sex,
            "age_approx": age_approx,
            "anatom_site_general_challenge": anatom_site_general_challenge,
        }
        # return {"image": image,
        #         "meta": meta,
        #         "target": target}
        return image, meta, target

    def __len__(self):
        return len(self.df)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

#todo: hair augment / microscope crop https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet

def get_data_transforms(cutout=True, cutout_length=16, size=224):

    mean = [0.7591, 0.5805, 0.5414]
    std = [0.0963, 0.1109, 0.1202]
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, valid_transform
