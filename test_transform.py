from train import read_mask, read_image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

transform = A.Compose([
    A.Rotate(limit=95, p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    # A.CoarseDropout(p=1, max_holes=3, max_height=8, max_width=8),
    # A.RandomBrightnessContrast(
    #     brightness_limit=(-0.9, -0.6),
    #     contrast_limit=0.2,
    #     brightness_by_max=False,
    #     p=1.0
    # ),
    A.RGBShift(
        r_shift_limit=10,  # Will sample red shift from [-30, 30]
        g_shift_limit=(-20, 20),  # Will sample green shift from [-20, 20]
        b_shift_limit=(-10, 10),  # Will sample blue shift from [-10, 10]
        p=1.0
    )
])


with np.printoptions(threshold=np.inf, linewidth=np.inf):
    x = read_image(
        'data/train/train/0bd55b1393e2ef89424de1556a26c8eb.jpeg', (100, 100))
    y = read_mask(
        'data/train_gt/train_gt/0bd55b1393e2ef89424de1556a26c8eb.jpeg', (100, 100))
    aug = transform(image=x, mask=y)
    x = aug['image']
    y = aug['mask']
    cv2.imwrite('img.jpeg', x)
    print(y)
    # x = cv2.imread('data/train_res/train_res/0aa5f7804c34a359bbb402345d341253.png')
    # res = x.sum(axis=(0, 1))
    # print(res)
