import numpy as np
import cv2
from tqdm import tqdm
import torch
from model import TResUnet
from utils import seeding
from train import load_data


def process_model_output(output_mask, target_size):
    assert output_mask.shape[0] == 1, 'Batch size (N) must be 1 during inference'
    assert output_mask.shape[1] == 3, 'Number of classes must be 3 (black, green, red)'

    output_mask = output_mask[0]

    predicted_mask = np.argmax(output_mask, axis=0)

    colors = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255)
    }

    colorized_mask = np.zeros(
        (predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    for label, color in colors.items():
        colorized_mask[predicted_mask == label] = color

    colorized_mask = cv2.resize(
        colorized_mask, target_size, interpolation=cv2.INTER_NEAREST)

    return colorized_mask

def evaluate(model, save_path, test_x, size):
    for x in tqdm(test_x):
        name = x.split("/")[-1].split('.')[0] + '.png'

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        original_shape = image.shape[:-1]
        image = cv2.resize(image, size)

        model_inp = np.transpose(image, (2, 0, 1))
        model_inp = model_inp/255.0
        model_inp = np.expand_dims(model_inp, axis=0)
        model_inp = model_inp.astype(np.float32)
        model_inp = torch.from_numpy(model_inp)
        model_inp = model_inp.to(device)

        with torch.no_grad():
            y_pred = model(model_inp)

        y_pred = process_model_output(
            y_pred.cpu(), (original_shape[1], original_shape[0]))
        cv2.imwrite(f'{save_path}/{name}', y_pred)


if __name__ == '__main__':
    seeding(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TResUnet()
    model = model.to(device)
    checkpoint_path = 'checkpoints/checkpoint_tres.pth'
    model.load_state_dict(torch.load(
        checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    path = 'data'
    (test_x, _), (_, _) = load_data(
        x_dir='data/test/test',
        val_ratio=0
    )

    test_x = sorted(test_x)
    print(len(test_x))

    save_path = 'data/test_res/test_res'
    size = (256, 256)
    evaluate(model, save_path, test_x, size)
