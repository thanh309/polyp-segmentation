import wandb
import os
import time
import datetime
import numpy as np
import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils import seeding, print_and_save, epoch_time
from model import TResUnet
# from model2 import RUPNet
from metrics import DiceBCELossMultipleClasses
from sklearn.model_selection import train_test_split

# Hyperparameters
image_size = 256
size = (image_size, image_size)
val_ratio = 0.2
batch_size = 8
num_epochs = 250
lr = 1e-4
early_stopping_patience = 20
lr_scheduler_patience = 5
lr_scheduler_factor = 0.5
aug_factor = 2
weight_decay = 1e-5

checkpoint_path = 'checkpoints/checkpoint_tres.pth'
# checkpoint_path = 'checkpoints/checkpoint_rup.pth'
data_path = 'data'

data_str = f'Image size: {size}\nBatch size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n'
data_str += f'Early stopping patience: {early_stopping_patience}\n'


def load_data(
        x_dir='data/train/train',
        y_dir='data/train_gt/train_gt',
        val_ratio=0.1
    ):

    train_files = os.listdir(x_dir)
    train_x = [os.path.join(x_dir, f) for f in train_files]
    if not val_ratio:
        return (train_x, None), (None, None)
    train_y = [os.path.join(y_dir, f) for f in train_files]

    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=val_ratio, random_state=42
    )

    return (train_x, train_y), (valid_x, valid_y)


def read_image(image_path, size):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, size)
    return image


def read_mask(mask_path, size):
    image = cv2.imread(mask_path)

    image = cv2.resize(image, size)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    red_mask = lower_mask + upper_mask
    red_mask[red_mask != 0] = 2

    green_mask = cv2.inRange(image, (36, 25, 25), (70, 255, 255))
    green_mask[green_mask != 0] = 1

    full_mask = cv2.bitwise_or(red_mask, green_mask)

    return full_mask


class BKAIIGHNeoDataset(Dataset):

    def __init__(self, images_path, masks_path, size, aug_factor=1, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        assert aug_factor > 0, 'aug_factor must be greater than 0'
        if aug_factor > 1:
            assert transform != None, 'transform must be != None for aug_factor > 1' 
        self.aug_factor = aug_factor
        self.n_samples = len(images_path) * aug_factor
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        original_idx = idx // self.aug_factor

        img_path = self.images_path[original_idx]
        mask_path = self.masks_path[original_idx]

        image = read_image(img_path, self.size)
        mask = read_mask(mask_path, self.size)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        image = image.astype('float32') / 255.
        image = image.transpose((2, 0, 1))

        mask = mask.astype('float32')

        return np.asarray(image), np.asarray(mask)



def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)

        return epoch_loss

if __name__ == '__main__':
    seeding(42)

    train_log_path = 'train_log.txt'
    if os.path.exists(train_log_path):
        print('NOTICE: old log file exists')
    else:
        train_log = open('train_log.txt', 'w')
        train_log.write('\n')
        train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print('\n\n\n')

    print_and_save(train_log_path, data_str)

    (train_x, train_y), (valid_x, valid_y) = load_data(val_ratio=val_ratio)
    # train_x = train_x[:100]
    # train_y = train_y[:100]



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


    train_dataset = BKAIIGHNeoDataset(train_x, train_y, size, aug_factor, transform=transform)
    valid_dataset = BKAIIGHNeoDataset(valid_x, valid_y, size)

    data_str = f'Dataset size:\nTrain: {len(train_dataset)} - Valid: {len(valid_dataset)}\n'
    print_and_save(train_log_path, data_str)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda')
    model = TResUnet()
    # model = RUPNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=lr_scheduler_patience, factor=lr_scheduler_factor
    )
    loss_fn = DiceBCELossMultipleClasses()
    loss_name = 'BCE Dice Loss'
    data_str = f'Optimizer: Adam\nLoss: {loss_name}\n'
    print_and_save(train_log_path, data_str)


    # Start training
    PROJECT = 'BKAI-IGH NeoPolyp'
    RESUME = 'allow'
    wandb.init(
        project=PROJECT,
        resume=RESUME,
        name=datetime_object,
        config={
            'image_size': image_size,
            'val_ratio': val_ratio,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate_init': lr,
            'early_stopping_patience': early_stopping_patience,
            'lr_scheduler_patience': lr_scheduler_patience,
            'lr_scheduler_factor': lr_scheduler_factor,
            'loss': loss_name,
            'aug_factor': aug_factor,
            'weight_decay': weight_decay
        }
    )
    wandb.watch(model)

    best_valid_loss = np.inf
    early_stopping_count = 0

    for epoch in range(num_epochs):
        curr_lr = scheduler.get_last_lr()[0]
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            data_str = f'Val. loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint at {checkpoint_path}...'
            print_and_save(train_log_path, data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_loss > best_valid_loss:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch time: {epoch_mins}m {epoch_secs}s | Current learning rate: {curr_lr}\n'
        data_str += f'\tTrain Loss: {train_loss:.4f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.4f}\n'
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'best_valid_loss': best_valid_loss,
            'learning_rate': curr_lr
        })
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f'Early stopping: validation loss stops improving from last {early_stopping_patience} epochs.\n'
            print_and_save(train_log_path, data_str)
            break
    
    wandb.finish()
