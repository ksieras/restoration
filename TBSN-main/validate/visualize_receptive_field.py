import sys
sys.path.append('..')
import argparse
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from validate.base_function import calculate_ssim
from util.build import build
from util.io import tensor2image
from util.option import parse, recursive_print
from torch.cuda.amp import autocast


def main(opt):
    validation_loaders = []
    for validation_dataset_opt in opt['validation_datasets']:
        ValidationDataset = getattr(__import__('dataset'), validation_dataset_opt['type'])
        validation_set = build(ValidationDataset, validation_dataset_opt['args'])
        validation_loader = DataLoader(validation_set, batch_size=1)
        validation_loaders.append(validation_loader)

    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt)
    model.data_parallel()

    network = model.networks['bsn']
    gradient, count = None, 0
    with autocast():
        for data in validation_loaders[0]:
            input = data['L']
            input = input[:, :, 64:192, 64:192]
            input.requires_grad = True
            output = network(input)
            center_output = torch.mean(output[:, :, 64, 64])
            center_output.backward()
            if gradient is None:
                gradient = torch.sum(torch.abs(input.grad), dim=1, keepdim=True).cpu()
            else:
                gradient += torch.sum(torch.abs(input.grad), dim=1, keepdim=True).cpu()
            count += 1

            if count == 50:
                break

    gradient = gradient / count * 10
    gradient = torch.clamp(gradient, 0, 1)
    gradient = gradient[0, 0].numpy()
    cam = cv2.applyColorMap(np.uint8(gradient * 255), cv2.COLORMAP_INFERNO)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cam)
    image.save('gradient_tbsn.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate the denoiser")
    parser.add_argument("--config_file", type=str, default='../option/tbsn.json')
    argspar = parser.parse_args()

    opt = parse(argspar.config_file)
    recursive_print(opt)

    main(opt)
