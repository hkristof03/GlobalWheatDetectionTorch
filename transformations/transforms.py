import os
import sys

import numpy as np
import torch

from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

p = os.path.abspath('.')
if p not in sys.path:
    print(p)
    sys.path.append(p)

from utils import config_parser


def to_tensor():

    return A.Compose([
        ToTensorV2(p=1.0)
        ],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )


class ImgAugTrainTransform:
    def __init__(self):
        self.transform = iaa.Sequential(
            [
                iaa.Sometimes(0.5,
                    iaa.SomeOf((1, 2), [
                        iaa.Fliplr(1.0),
                        iaa.Flipud(1.0),
                    ])
                ),
                iaa.OneOf([
                    iaa.Sometimes(0.3, [
                        iaa.OneOf([
                            iaa.Multiply((0.7, 1.2)),
                            iaa.MultiplyElementwise((0.7, 1.2)),
                        ]),
                        iaa.OneOf([
                            iaa.MultiplySaturation((5.0, 10.0)), # good
                            iaa.MultiplyHue((1.5, 3.0)),
                            iaa.LinearContrast((0.8, 2.0)),
                            iaa.AllChannelsHistogramEqualization(),
                        ]),
                    ]),
                    iaa.Sometimes(0.3, [
                        iaa.SomeOf((1, 2), [
                            iaa.pillike.EnhanceColor((1.1, 1.6)),
                            iaa.pillike.EnhanceSharpness((0.7, 1.6)),
                            iaa.pillike.Autocontrast(cutoff=(4, 8)),
                            iaa.MultiplySaturation((1.2, 5.1)),
                        ])
                    ])
                ]),
                iaa.Sometimes(0.3, [
                    iaa.Dropout(p=(0.01, 0.09)),
                    iaa.GaussianBlur((0.4, 1.5)),
                ]),
            ],
            random_order=True # apply the augmentations in random order
        )

    def __call__(self, image, bounding_boxes):

        bboxes = [BoundingBox(*bbox) for bbox in bounding_boxes]
        bboxes = BoundingBoxesOnImage(bboxes, image.shape)
        image_aug, bboxes_aug = self.transform(
            image=image,
            bounding_boxes=bboxes
        )
        image_aug = image_aug.astype(np.float32)
        image_aug /= 255.0
        image_aug = torch.tensor(image_aug).permute(2,0,1)
        bboxes_aug = torch.tensor(
            bboxes_aug.to_xyxy_array(),
            dtype=torch.float32
        )

        return image_aug, bboxes_aug
