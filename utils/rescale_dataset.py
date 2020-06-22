import os
from ast import literal_eval
import argparse

import pandas as pd
import numpy as np

from PIL import Image

from tqdm import tqdm

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def parse_args():
    parser = argparse.ArgumentParser(
        description='Provide list of image sizes for rescale."
    )
    parser.add_argument('-sizes', '--list', nargs='+', 
        help='List of desired image sizes.', default=[256, 512]
    )
    return parser.parse_args()


def rescale_dataset(path, path_save, size):
    """
    """
    path_save = os.path.join(*path_save)
    
    if not os.path.isdir(path_save):
        os.mkdir(path_save)
        
    df_res = pd.DataFrame()
    
    seq = iaa.Sequential([
            iaa.size.Resize((size, size)),
        ])
    df = pd.read_csv(os.path.join(*path,'train.csv'))

    df['bbox'] = df['bbox'].apply(lambda x: literal_eval(x))
    # imgaug bbox requires top left and bottom right coordinates, while the competition
    # coordinates are in the form of [xmin, ymin, width, height]
    bboxes = list(df['bbox'])
    imgaug_boxes = []

    for bbox in bboxes:
        xmin, ymin, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        imgaug_boxes.append([xmin, ymin, xmin+width, ymin+height])

    df['imgaug_bbox'] = imgaug_boxes

    image_ids = list(df['image_id'].unique())
    
    for i, img_id in enumerate(tqdm(image_ids)):

        bboxes = list(df.loc[(df['image_id'] == img_id)]['imgaug_bbox'])

        img = Image.open(os.path.join(*path, 'train', img_id + '.jpg'))
        img = np.array(img)

        bboxes_ = []

        for bbox in bboxes:
            bboxes_.append(BoundingBox(*bbox))

        bbs = BoundingBoxesOnImage(bboxes_, shape=img.shape)
        
        image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
        image_aug = Image.fromarray(image_aug)
        image_aug.save(os.path.join(path_save, img_id + '.jpg'))

        width = df.loc[(df['image_id'] == img_id)]['width'].values[0]
        height = df.loc[(df['image_id'] == img_id)]['height'].values[0]
        source = df.loc[(df['image_id'] == img_id)]['source'].values[0]

        bboxes = bbs_aug.to_xyxy_array()
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

        length = len(bboxes)

        d = {
            'image_id': [img_id for i in range(length)],
            'width': [width for i in range(length)],
            'height': [height for i in range(length)],
            'source': [source for i in range(length)],
            'bbox': [list(bbox) for bbox in bboxes],
        }

        df_ = pd.DataFrame(d)
        df_res = pd.concat([df_res, df_], axis=0, sort=True)
        
    df_res.to_csv(os.path.join(*path, f'train_{size}x{size}.csv'), index=False)
    
    
if __name__ == '__main__':
    
    args = parse_args()
    sizes = args.sizes
    path = ['..','datasets']

    for i, size in enumerate(tqdm(sizes)):
        
        path_save = path + [f'train_{size}x{size}']
        rescale_dataset(path, path_save, size)