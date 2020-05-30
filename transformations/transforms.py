import os
import sys

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

p = os.path.abspath('.')
if p not in sys.path:
    print(p)
    sys.path.append(p)

from utils import config_parser

# Albumentations
def get_train_transform():

    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():

    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
