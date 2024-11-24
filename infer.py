import argparse
import torch
from test import eval_single
from model import TResUnet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a single image.")
    parser.add_argument('--image_path', type=str,
                        required=True, help='Path to the input image.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_tres78798.pth',
                        help='Path to the trained model checkpoint. Default="checkpoints/checkpoint_tres78798.pth"')
    parser.add_argument('--output_path', type=str, default='output.png',
                        help='Path to save the segmented output image. Default="output.png"')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                        else 'cpu', help='Device to run inference on. Default based on cuda availability')
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    model = TResUnet()
    model = model.to(args.device)
    checkpoint_path = args.checkpoint
    model.load_state_dict(torch.load(
        checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    x = args.image_path
    output_path = args.output_path
    size = (256, 256)

    eval_single(model, output_path, x, size, device)
    print(f'Output saved at {output_path}')


if __name__ == "__main__":
    main()
