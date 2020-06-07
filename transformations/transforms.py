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


class ImgAugTrainTransform:
    def __init__(self):
        self.aug = seq = iaa.Sequential(
            [
                #iaa.AllChannelsHistogramEqualization(),
               # iaa.MultiplySaturation((0.5, 1.5)),
                iaa.pillike.EnhanceColor((1.5, 1.6)),
                iaa.pillike.EnhanceSharpness((1.5, 1.6)),
                iaa.pillike.Autocontrast(cutoff=(12, 12)),
                iaa.MultiplySaturation((1.2, 1.4)),
                ToTensorV2(p=1.0),
            ],
            random_order=False
        ) # apply augmenters in random order

    def __call__(self, img):
        return self.aug.augment_image(img)
