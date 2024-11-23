from train import read_mask, read_image
import numpy as np
import albumentations as A
import cv2

transform =  A.Compose([
    A.Rotate(limit=45, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.4),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=False,
        p=0.5
    ),
    A.RGBShift(
        r_shift_limit=5,
        g_shift_limit=10,
        b_shift_limit=5,
        p=0.5
    ),
    # A.CoarseDropout(p=0.3, max_holes=10, max_height=24, max_width=24)
])

with np.printoptions(threshold=np.inf, linewidth=np.inf):
    _ = cv2.imread('data/train/train/0bd55b1393e2ef89424de1556a26c8eb.jpeg')
    shape = _.shape[::-1][1:]
    x = read_image(
        'data/train/train/0bd55b1393e2ef89424de1556a26c8eb.jpeg', shape)
    y = read_mask(
        'data/train_gt/train_gt/0bd55b1393e2ef89424de1556a26c8eb.jpeg', shape)
    aug = transform(image=x, mask=y)
    x = aug['image']
    y = aug['mask']
    cv2.imwrite('img.jpeg', x)
    # print(y)
    # x = cv2.imread('data/train_res/train_res/0aa5f7804c34a359bbb402345d341253.png')
    # res = x.sum(axis=(0, 1))
    # print(res)
