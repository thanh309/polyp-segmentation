from train import read_mask, read_image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

transform =  A.Compose([
    A.Rotate(limit=95, p=1),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.CoarseDropout(p=1, max_holes=3, max_height=8, max_width=8),
])




with np.printoptions(threshold=np.inf, linewidth=np.inf):
    # x = read_image('data/train/train/0bd55b1393e2ef89424de1556a26c8eb.jpeg', (100, 100))
    # y = read_mask('data/train_gt/train_gt/0bd55b1393e2ef89424de1556a26c8eb.jpeg', (100, 100))
    # aug = transform(image=x, mask=y)
    # x = aug['image']
    # y = aug['mask']
    # cv2.imwrite('img.jpeg', x)
    # print(y)
    x = cv2.imread('data/train_res/train_res/0aa5f7804c34a359bbb402345d341253.png')
    res = x.sum(axis=(0, 1))
    print(res)
