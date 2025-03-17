import os
import numpy as np
from PIL import Image
import numbers
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
from torchvision.datasets.folder import has_file_allowed_extension
import sys
import json
from pprint import pprint
from time import time
from dataset.base_function import  crop_3
__all__ = ['fluore_to_tensor', 'DenoisingFolder', 'DenoisingFolderN2N', 
           'DenoisingTestMixFolder', 'load_denoising', 
           'load_denoising_n2n_train', 'load_denoising_test_mix']

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def pil_loader(path):
    img = Image.open(path)
    img_np = np.asarray(img)
    img.close()
    img_np = np.expand_dims(img_np,axis=0)
    #print('fmd\'shape is ',img_np.shape)
    #img_np = np.transpose(img_np, (2, 0, 1))
    return img_np


def fluore_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. Range stays the same.
    Only output one channel, if RGB, convert to grayscale as well.
    Currently data is 8 bit depth.
    
    Args:
        pic (PIL Image): Image to be converted to Tensor.
    Returns:
        Tensor: only one channel, Tensor type consistent with bit-depth.
    """
    if not(_is_pil_image(pic)):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        # all 8-bit: L, P, RGB, YCbCr, RGBA, CMYK
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    img = img.view(pic.size[1], pic.size[0], nchannel)
    
    if nchannel == 1:
        img = img.squeeze(-1).unsqueeze(0)
    elif pic.mode in ('RGB', 'RGBA'):
        # RBG to grayscale: 
        # https://en.wikipedia.org/wiki/Luma_%28video%29
        ori_dtype = img.dtype
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        img = (img[:, :, [0, 1, 2]].float() * rgb_weights).sum(-1).unsqueeze(0)
        img = img.to(ori_dtype)
    else:
        # other type not supported yet: YCbCr, CMYK
        raise TypeError('Unsupported image type {}'.format(pic.mode))

    return img


class DenoisingFolder(torch.utils.data.Dataset):
    """Class for the denoising dataset for both train and test, with 
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples

    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int, optional): default 19. 19th fov is test fov
        captures (int): select # images within one folder
        transform (callable, optional): A function/transform that takes in 
            an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes 
            in the target and transforms it.
        loader (callable, optional): image loader
    """
    def __init__(self, root, train, noise_levels,patch_size, types=None, test_fov=19,
        captures=50, transform=None, target_transform=None, loader=pil_loader):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]
        if train==True:
            all_types = ['TwoPhoton_MICE', 'Confocal_MICE','Confocal_FISH',
            'TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B', 'Confocal_BPAE_R',
            'Confocal_BPAE_G', 'Confocal_BPAE_B', 
            'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        else:
            all_types = [  'TwoPhoton_MICE'] #'Confocal_MICE'] #,'Confocal_FISH','TwoPhoton_MICE']
        assert all([level in all_noise_levels for level in noise_levels])
        self.noise_levels = all_noise_levels  #noise_levels
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        self.patch_size = patch_size
        self.root = root
        self.train = train
        if train:
            fovs = list(range(1, 20+1))
            fovs.remove(test_fov)
            self.fovs = fovs
        else:
            self.fovs = [test_fov]
        self.captures = captures
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'train' if train else 'test',
                        'Noise levels': self.noise_levels,
                        f'{len(self.types)} Types': self.types,
                        'Fovs': self.fovs,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        # types: microscopy_cell
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]

        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            for noise_level in self.noise_levels:
                if noise_level == 1:
                    noise_dir = os.path.join(subdir, 'raw')
                elif noise_level in [2, 4, 8, 16]:
                    noise_dir = os.path.join(subdir, f'avg{noise_level}')
                for i_fov in self.fovs:
                    noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
                    clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                    if self.train:
                        noisy_captures = []
                        for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                            if is_image_file(fname):
                                noisy_file = os.path.join(noisy_fov_dir, fname)
                                noisy_captures.append(noisy_file)
                        # randomly select one noisy capture when loading from FOV     
                        samples.append((noisy_captures, clean_file))
                    else:
                        # for test, only one FOV, use all of them
                        print('load test data folder:',noisy_fov_dir)
                        for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                            if is_image_file(fname) :
                                noisy_file = os.path.join(noisy_fov_dir, fname)
                                samples.append((noisy_file, clean_file))

        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        if self.train:
            noisy_captures, clean_file = self.samples[index]
            idx = np.random.choice(len(noisy_captures), 1)
            noisy_file = noisy_captures[idx[0]]
        else:
            noisy_file, clean_file = self.samples[index]
            
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        #print('begin transform............')
        if self.transform is not None:
            #print('transform is not None')
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)
        #print('begin crop operation,shape:',noisy.shape)
        img_L, img_H = crop_3(self.patch_size, noisy, clean)
        #print('after crop operation,shape:',img_L.shape)
        img_L, img_H = np.float32(img_L) / 255., np.float32(img_H) / 255.
        #print('begin from_numpy operation')
        img_L, img_H = torch.from_numpy(img_L), torch.from_numpy(img_H)
        #print('repeat............train data shape:',img_L.shape)
        img_L = img_L.repeat(3,1,1)
        img_H = img_H.repeat(3,1,1)
        #print('after repeat shape:',img_L.shape)
        return {'L':img_L,'H': img_H}

    def __len__(self):
        return len(self.samples)


class DenoisingFolderN2N(torch.utils.data.Dataset):
    """Data loader for denoising dataset for only train, Noise2Noise!
    with file structure:
        data_root/type/noise_level/fov/captures.png
    For test specific type, use DenoisingFolder
    For test mixed types, use DenoisingFolderTestMix
    Read in all 50 captures, but randomly select 2 during the training.
    Only consider the same noise level for the input and target.
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        captures.png:   50 images in each fov --> use fewer samples
    Args:
        train (bool): Training set if True, else Test set
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
        types (seq): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int): default 19. 19th fov is test fov
        captures (int): # images within one folder
    """
    def __init__(self, root, noise_levels, types=None, test_fov=19,
        captures=50, transform=None, target_transform=None, loader=pil_loader):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]
        all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
            'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
            'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
            'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        assert all([level in all_noise_levels for level in noise_levels])
        self.noise_levels = noise_levels
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        self.root = root
        fovs = list(range(1, 20+1))
        fovs.remove(test_fov)
        self.fovs = fovs
       
        self.captures = captures
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'train N2N',
                        'Noise levels': self.noise_levels,
                        f'{len(self.types)} Types': self.types,
                        'Fovs': self.fovs,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        # types: microscopy_cell
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]

        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            for noise_level in self.noise_levels:
                if noise_level == 1:
                    noise_dir = os.path.join(subdir, 'raw')
                elif noise_level in [2, 4, 8, 16]:
                    noise_dir = os.path.join(subdir, f'avg{noise_level}')
                for i_fov in self.fovs:
                    noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
                    clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                    noisy_captures = []
                    for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                        if is_image_file(fname):
                            noisy_captures.append(os.path.join(noisy_fov_dir, fname))
                    # noisy_captures contains a list of all captures
                    # randomly select 2 when loading
                    samples.append((noisy_captures, clean_file))
        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy_input, noisy_target, clean)
        """
        noisy_files, clean_file = self.samples[index]
        idx = np.random.choice(len(noisy_files), 2, replace=False)
        noisy_input = self.loader(noisy_files[idx[0]])
        noisy_target = self.loader(noisy_files[idx[1]])
        clean = self.loader(clean_file)
        if self.transform is not None:
            noisy_input = self.transform(noisy_input)
            noisy_target = self.transform(noisy_target)
        if self.target_transform is not None:
            clean = self.target_transform(clean)

        return noisy_input, noisy_target, clean

    def __len__(self):
        return len(self.samples)


class DenoisingTestMixFolder(torch.utils.data.Dataset):
    """Data loader for the denoising mixed test set.
        data_root/test_mix/noise_level/imgae.png
        type:           test_mix
        noise_level:    5 (+ 1: ground truth)
        captures.png:   48 images in each fov
    Args:
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
    """

    def __init__(self, root, loader, noise_levels, transform, target_transform):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16] 
    
        assert all([level in all_noise_levels for level in all_noise_levels])
        self.noise_levels = noise_levels

        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'test_mix',
                        'Noise levels': self.noise_levels,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))


    
    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        test_mix_dir = os.path.join(root_dir, 'test_mix')
        gt_dir = os.path.join(test_mix_dir, 'gt')
        test_type = 'Confocal_FISH' #Confocal_MICE Confocal_FISH,TwoPhoton_MICE
        print('the test type :',test_type)
        for noise_level in self.noise_levels:
            if noise_level == 1:
                noise_dir = os.path.join(test_mix_dir, 'raw')
            elif noise_level in [2, 4, 8, 16]:
                noise_dir = os.path.join(test_mix_dir, f'avg{noise_level}')

            for fname in sorted(os.listdir(noise_dir)):
                if is_image_file(fname) and test_type in fname:
                    noisy_file = os.path.join(noise_dir, fname)
                    clean_file = os.path.join(gt_dir, fname)
                    samples.append((noisy_file, clean_file))
        print(samples)
        return samples


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)
        img_L,img_H = noisy,clean
        img_L, img_H = np.float32(img_L) / 255., np.float32(img_H) / 255.
        #print('test_data-begin from_numpy operation')
        #print('img_L shape:{},img_H shape:{}'.format(img_L.shape,img_H.shape))
        img_L,img_H = torch.from_numpy(img_L),torch.from_numpy(img_H)
        img_L = img_L.repeat(3,1,1)
        img_H = img_H.repeat(3,1,1)
        #print('test_data-after repeat shape:',img_L.shape)
        return {'L':img_L,'H': img_H}


    def __len__(self):
        return len(self.samples)


def load_denoising(root, train, batch_size, noise_levels, types=None, captures=2,
    patch_size=256, transform=None, target_transform=None, loader=pil_loader,
    test_fov=19):
    """
    files: root/type/noise_level/fov/captures.png
        total 12 x 5 x 20 x 50 = 60,000 images
        raw: 12 x 20 x 50 = 12,000 images
    
    Args:
        root (str): root directory to dataset
        train (bool): train or test
        batch_size (int): e.g. 4
        noise_levels (seq): e.g. [1, 2, 4], or [1, 2, 4, 8]
        types (seq, None): e.g. [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    #if transform is None:
    #    # default to center crop the image from 512x512 to 256x256
    #    transform = transforms.Compose([
    #        transforms.CenterCrop(patch_size),
    #        fluore_to_tensor,
    #        transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
    #        ])
    #target_transform = transform
        
    dataset = DenoisingFolder(root, train, noise_levels,patch_size, 
        types=types, test_fov=test_fov,
        captures=captures, transform=transform, 
        target_transform=target_transform, loader=pil_loader)
    print('dataset preperation over...........')
    kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, **kwargs)

    return data_loader
    

def load_denoising_n2n_train(root, batch_size, noise_levels, types=None,
    patch_size=256, transform=None, target_transform=None, loader=pil_loader,
    test_fov=19):
    """For N2N model, use all captures in each fov, randomly select 2 when
    loading.
    files: root/type/noise_level/fov/captures.png
        total 12 x 5 x 20 x 50 = 60,000 images
        raw: 12 x 20 x 50 = 12,000 images
    
    Args:
        root (str):
        batch_size (int): 
        noise_levels (seq): e.g. [1, 2, 4], or [1, 2, 4, 8]
        types (seq, None): e.g.     [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    if transform is None:
        # default to center crop the image from 512x512 to 256x256
        transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            fluore_to_tensor,
            transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
            ])
    target_transform = transform
    dataset = DenoisingFolderN2N(root, noise_levels, types=types, 
        test_fov=test_fov, transform=transform, 
        target_transform=target_transform, loader=pil_loader)
    kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, **kwargs)

    return data_loader


def load_denoising_test_mix(root, batch_size, noise_levels, loader=pil_loader, 
    transform=None, target_transform=None, patch_size=256):
    """
    files: root/test_mix/noise_level/captures.png
        
    Args:
        root (str):
        batch_size (int): 
        noise_levels (seq): e.g. [1, 2, 4], or [1, 2, 4, 8]
        types (seq, None): e.g.     [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    #if transform is None:
    #    transform = transforms.Compose([
    #        transforms.CenterCrop(patch_size),
    #        fluore_to_tensor,
    #        transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
    #        ])
    # the same
    #print('data load batch:',batch_size)
    target_transform = transform
    dataset = DenoisingTestMixFolder(root, loader, noise_levels, transform, 
        target_transform)
    kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, **kwargs)

    return data_loader





#if __name__ == 'main':
def data_load(train):
    root = '/data/fmdd'
    loader = pil_loader
    #train = False
    noise_levels = [1, 2, 4]
    types = None  #['TwoPhoton_MICE']
    # types = None
    captures = 10
    patch_size = 120
    if train == True:
        batch_size = 16
    else:
        batch_size = 1
    transform = None
    target_transform = None
    test_fov = 19
    tic = time()
    # dataset = DenoisingFolder(root, train, noise_levels, types=types, test_fov=19,
    #     captures=2, transform=None, target_transform=None, loader=pil_loader)
    # print(time()-tic)
    # print(dataset.samples[0])
    # print(dataset[0])
    kwargs = {'drop_last': False}
    add_kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    kwargs.update(add_kwargs)

    loader = load_denoising(root, train, batch_size, noise_levels=noise_levels,
        types=types, captures=captures, patch_size=patch_size, transform=transform, 
        target_transform=target_transform, loader=pil_loader, test_fov=test_fov)
    return loader
    #print('data_loader running successfull')
    '''for batch_size, (noisy, clean) in enumerate(loader):
        print(noisy.shape)
        print(clean.shape)
        break'''
   # print('train_data load successfully')
   # transform = transforms.Compose([
   #     transforms.FiveCrop(patch_size),
   #     transforms.Lambda(lambda crops: torch.stack([
   #         fluore_to_tensor(crop) for crop in crops])),
   #     transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
   #     ])
    #batch_size = 1
    #test_loader = load_denoising_test_mix(root, batch_size, noise_levels=noise_levels, loader=pil_loader, transform=transform, )

    #print(len(test_loader.dataset))
    #print(test_loader.dataset.samples[0])
    #for batch_size, (noisy, clean) in enumerate(test_loader):
    #    print('nosiy in test_loader:',noisy.shape)
    #    break
    #if train:
    #    print('train data return')
    #    return loader
    #else:
    #    print('test data return..................................')
    #    return test_loader
    '''train_loader = load_denoising_n2n_train(root, batch_size, noise_levels, types=None,
    patch_size=256, transform=transform, target_transform=transform, loader=pil_loader,
    test_fov=19)

    for batch_size, (noisy_input, noisy_target, clean) in enumerate(train_loader):
        print(noisy_input.shape)
        print(noisy_target.shape)
        print(clean.shape)
        break'''
