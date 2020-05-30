from ast import literal_eval
from typing import Tuple, Callable
import random
random.seed(2020)

import torch
from torch.utils.data import DataLoader, Dataset

import cv2
import pandas as pd
import numpy as np

from utils.config_parser import parse_args, parse_yaml
from transformations import transforms

DIR_INPUT = './datasets'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index):

        image_id = self.image_ids[index]
        records = self.df.loc[(self.df['image_id'] == image_id)]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(
                tuple(map(torch.tensor, zip(*sample['bboxes'])))
            ).permute(1, 0)

        return image, target, image_id

    def __len__(self):

        return self.image_ids.shape[0]


def get_train_valid_df(
    path_df: str,
    valid_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    """
    df = pd.read_csv(path_df)

    box_data = np.stack(df['bbox'].apply(lambda x: literal_eval(x)))
    df[['x', 'y', 'w', 'h']] = pd.DataFrame(box_data).astype(np.float)

    image_ids = df['image_id'].unique()
    random.shuffle(image_ids)
    valid_size = round(len(image_ids) * valid_size)

    train_ids = image_ids[:-valid_size]
    valid_ids = image_ids[-valid_size:]

    train_df = df.loc[(df['image_id'].isin(train_ids))].copy()
    valid_df = df.loc[(df['image_id'].isin(valid_ids))].copy()

    return train_df, valid_df

def collate_fn(batch):
    print(f"Batch's type: {type(batch)}, Batch's shape: {len(batch)}")
    return tuple(zip(*batch))

def get_train_valid_dataloaders(
    config_dataloader: dict,
    collate_fn: Callable,
    ) -> Tuple[DataLoader, DataLoader]:
    """"""
    path_df = config_dataloader['path_df']
    dir_train = config_dataloader['train_dataset']['dir_train']

    train_df, valid_df = get_train_valid_df(path_df)
    train_trf = transforms.get_train_transform()
    valid_trf = transforms.get_valid_transform()

    train_dataset = WheatDataset(
        train_df,
        dir_train,
        train_trf,
    )
    valid_dataset = WheatDataset(
        valid_df,
        dir_train,
        valid_trf,
    )
    config_train_loader = config_dataloader['train_loader']
    config_valid_loader = config_dataloader['valid_loader']
    train_data_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        **config_train_loader
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        collate_fn=collate_fn,
        **config_valid_loader
    )
    return (train_data_loader, valid_data_loader)


if __name__ == '__main__':

    args = parse_args()
    configs = parse_yaml(args.pyaml)
    configs_dataloader = configs['dataloader']

    train_data_loader, valid_data_loader = get_train_valid_dataloaders(
        configs_dataloader,
        collate_fn
    )
    print(len(train_data_loader))
