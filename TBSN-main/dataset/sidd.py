import sys
sys.path.append('..')
from dataset.base_function import dataset_path, crop_3
import glob
import imageio
import numpy as np
import os
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sidd_path = os.path.join(dataset_path, 'SIDD')

def open_image_SIDD(path):
    img = Image.open(path)
    img_np = np.asarray(img)
    img.close()
    print('SIDD\' shape is',img_np.shape)
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np

class SIDDMediumTrainDataset(Dataset):
    def __init__(self, pin_memory, patch_size):
        super().__init__()
        self.pin_memory = pin_memory
        self.patch_size = patch_size

        self._img_paths = self._get_img_paths()
        if self.pin_memory:
            self._imgs = self._open_images()

    def __getitem__(self, index):
        index = index % len(self._img_paths)

        if self.pin_memory:
            img_L = self._imgs[index]['L']
            img_H = self._imgs[index]['H']
        else:
            img_path = self._img_paths[index]
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])

        img_L, img_H = crop_3(self.patch_size, img_L, img_H)
        img_L, img_H = np.float32(img_L) / 255., np.float32(img_H) / 255.
        img_L, img_H = torch.from_numpy(img_L), torch.from_numpy(img_H)

        return {'L': img_L, 'H': img_H}

    def __len__(self):
        length = len(self._img_paths)
        if length <= 10000 and length!=0:
            return (10000 // length) * length
        else:
            return length

    def _get_img_paths(self):
        img_paths = []
        print('root directory is ',sidd_path)
        L_pattern = os.path.join(sidd_path, 'SIDD_Medium_Srgb/Data/*/NOISY_SRGB_*.PNG')
        L_paths = sorted(glob.glob(L_pattern))

        print(L_paths)
        for L_path in L_paths:
            img_paths.append({'L': L_path, 'H': L_path.replace('NOISY', 'GT')})
        return img_paths

    def _open_images(self):
        imgs = []
        for img_path in self._img_paths:
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])
            imgs.append({'L': img_L, 'H': img_H})
        return imgs

    def _open_image(self, path):
        return open_image_SIDD(path)


class SIDDValidationDataset(Dataset):
    def __init__(self):
        super().__init__()
        self._open_images(sidd_path)
        self.n = self.noisy_block.shape[0]
        self.k = self.noisy_block.shape[1]

    def __getitem__(self, index):
        index_n = index // self.k
        index_k = index % self.k

        img_H = self.gt_block[index_n, index_k]
        img_H = np.float32(img_H) / 255.
        img_H = np.transpose(img_H, (2, 0, 1))
        img_H = img_H

        img_L = self.noisy_block[index_n, index_k]
        img_L = np.float32(img_L) / 255.
        img_L = np.transpose(img_L, (2, 0, 1))
        img_L = img_L

        return {'H':img_H, 'L':img_L}

    def __len__(self):
        return self.n * self.k

    def _open_images(self, path):
        mat = sio.loadmat(os.path.join(path, 'SIDD_Validation/ValidationNoisyBlocksSrgb.mat'))
        self.noisy_block = mat['ValidationNoisyBlocksSrgb']
        mat = sio.loadmat(os.path.join(path, 'SIDD_Validation/ValidationGtBlocksSrgb.mat'))
        self.gt_block = mat['ValidationGtBlocksSrgb']


class SIDDBenchmarkDataset(Dataset):
    def __init__(self):
        super().__init__()
        self._open_images(sidd_path)
        self.n = self.noisy_block.shape[0]
        self.k = self.noisy_block.shape[1]

    def __getitem__(self, index):
        index_n = index // self.k
        index_k = index % self.k

        img_L = self.noisy_block[index_n, index_k]
        img_L = np.float32(img_L) / 255.
        img_L = np.transpose(img_L, (2, 0, 1))
        img_L = img_L

        return {'L': img_L}

    def __len__(self):
        return self.n * self.k

    def _open_images(self, path):
        mat = sio.loadmat(os.path.join(path, 'SIDD_Benchmark/BenchmarkNoisyBlocksSrgb.mat'))
        self.noisy_block = mat['BenchmarkNoisyBlocksSrgb']


# extract sidd validation images
if __name__ == '__main__':
    dest_dir = os.path.join(sidd_path, 'SIDD_Validation/ValidationGTImagesSrgb')
    os.makedirs(dest_dir, exist_ok=True)

    mat = sio.loadmat(os.path.join(sidd_path, 'SIDD_Validation/ValidationNoisyBlocksSrgb.mat'))
    noisy_block = mat['ValidationNoisyBlocksSrgb']
    mat = sio.loadmat(os.path.join(sidd_path, 'SIDD_Validation/ValidationGtBlocksSrgb.mat'))
    gt_block = mat['ValidationGtBlocksSrgb']
    n = noisy_block.shape[0]
    k = noisy_block.shape[1]
    for i in tqdm(range(n)):
        for j in range(k):
            gt_img = gt_block[i, j]
            imageio.imwrite(os.path.join(dest_dir, '%02d_%02d.png' % (i, j)), gt_img)
