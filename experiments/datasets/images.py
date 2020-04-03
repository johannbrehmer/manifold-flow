import os
import logging
import numpy as np
import zipfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvt
from torchvision import datasets

from experiments.datasets.base import BaseSimulator
from experiments.datasets.utils import download_file_from_google_drive
from manifold_flow.training import UnlabelledImageFolder

logger = logging.getLogger(__name__)


class CIFAR10Fast(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

        self.data = self.data.transpose((0, 3, 1, 2))  # HWC -> CHW.
        self.data = torch.from_numpy(self.data)  # Shouldn't make a copy.
        assert self.data.dtype == torch.uint8

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Don't convert to PIL Image, just to convert back later: slow.

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageNet32(UnlabelledImageFolder):
    GOOGLE_DRIVE_FILE_ID = "1TXsg8TP5SfsSL6Gk39McCkZu9rhSQnNX"
    UNZIPPED_DIR_NAME = "imagenet32"
    UNZIPPED_TRAIN_SUBDIR = "train_32x32"
    UNZIPPED_VAL_SUBDIR = "valid_32x32"

    def __init__(self, root, train=True, download=False, transform=None):
        if download:
            self._download(root)

        img_dir = "train" if train else "val"
        super(ImageNet32, self).__init__(os.path.join(root, img_dir), transform=transform)

    def _download(self, root):
        if os.path.isdir(os.path.join(root, "train")):
            return  # Downloaded already

        os.makedirs(root, exist_ok=True)

        zip_file = os.path.join(root, self.UNZIPPED_DIR_NAME + ".zip")

        logger.info("Downloading {}...".format(os.path.basename(zip_file)))
        download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID, zip_file)

        logger.info("Extracting {}...".format(os.path.basename(zip_file)))
        with zipfile.ZipFile(zip_file, "r") as fp:
            fp.extractall(root)
        os.remove(zip_file)

        os.rename(os.path.join(root, self.UNZIPPED_DIR_NAME, self.UNZIPPED_TRAIN_SUBDIR), os.path.join(root, "train"))
        os.rename(os.path.join(root, self.UNZIPPED_DIR_NAME, self.UNZIPPED_VAL_SUBDIR), os.path.join(root, "val"))
        os.rmdir(os.path.join(root, self.UNZIPPED_DIR_NAME))


class ImageNet64(ImageNet32):
    GOOGLE_DRIVE_FILE_ID = "1NqpYnfluJz9A2INgsn16238FUfZh9QwR"
    UNZIPPED_DIR_NAME = "imagenet64"
    UNZIPPED_TRAIN_SUBDIR = "train_64x64"
    UNZIPPED_VAL_SUBDIR = "valid_64x64"


class ImageNet64Fast(Dataset):
    GOOGLE_DRIVE_FILE_ID = {"train": "15AMmVSX-LDbP7LqC3R9Ns0RPbDI9301D", "valid": "1Me8EhsSwWbQjQ91vRG1emkIOCgDKK4yC"}

    NPY_NAME = {"train": "train_64x64.npy", "valid": "valid_64x64.npy"}

    def __init__(self, root, train=True, download=False, transform=None):
        self.transform = transform
        self.root = root

        if download:
            self._download()

        tag = "train" if train else "valid"
        npy_data = np.load(os.path.join(root, self.NPY_NAME[tag]))
        self.data = torch.from_numpy(npy_data)  # Shouldn't make a copy.

    def __getitem__(self, index):
        img = self.data[index, ...]

        if self.transform is not None:
            img = self.transform(img)

        # Add a bogus label to be compatible with standard image data.
        return img, torch.tensor([0.0])

    def __len__(self):
        return self.data.shape[0]

    def _download(self):
        os.makedirs(self.root, exist_ok=True)

        for tag in ["train", "valid"]:
            npy = os.path.join(self.root, self.NPY_NAME[tag])
            if not os.path.isfile(npy):
                logger.info("Downloading {}...".format(self.NPY_NAME[tag]))
                download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID[tag], npy)


class CelebA(UnlabelledImageFolder):
    """Unlabelled standard CelebA dataset, the aligned version."""

    GOOGLE_DRIVE_FILE_ID = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    ZIP_FILE_NAME = "img_align_celeba.zip"

    def __init__(self, root, train=True, transform=None, download=False):
        if download:
            self.download(root)

        tag = "train" if train else "valid"
        super(CelebA, self).__init__(os.path.join(root, self.img_dir), transform=transform)

    def download(self, root):
        if os.path.isdir(os.path.join(root, self.img_dir)):
            return  # Downloaded already

        os.makedirs(root, exist_ok=True)

        zip_file = os.path.join(root, self.ZIP_FILE_NAME)

        print("Downloading {}...".format(os.path.basename(zip_file)))
        download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID, zip_file)

        print("Extracting {}...".format(os.path.basename(zip_file)))
        with zipfile.ZipFile(zip_file, "r") as fp:
            fp.extractall(root)

        os.remove(zip_file)


class CelebAHQ(CelebA):
    """Unlabelled high quality CelebA dataset with 256x256 images."""

    GOOGLE_DRIVE_FILE_ID = "1psLniAvAvyDgJV8DBk7cvTZ9EasB_2tZ"
    ZIP_FILE_NAME = "celeba-hq-256.zip"

    def __init__(self, root, transform=None, train=True, download=False):
        self.train = train
        super().__init__(root, transform=transform, download=download)

    @property
    def img_dir(self):
        if self.train:
            return "celeba-hq-256/train-png"
        else:
            return "celeba-hq-256/validation-png"


class CelebAHQ64Fast(Dataset):
    GOOGLE_DRIVE_FILE_ID = {"train": "1bcaqMKWzJ-2ca7HCQrUPwN61lfk115TO", "valid": "1WfE64z9FNgOnLliGshUDuCrGBfJSwf-t"}

    NPY_NAME = {"train": "train.npy", "valid": "valid.npy"}

    def __init__(self, root, train=True, download=False, transform=None):
        self.transform = transform
        self.root = root

        if download:
            self._download()

        tag = "train" if train else "valid"
        npy_data = np.load(os.path.join(root, self.NPY_NAME[tag]))
        self.data = torch.from_numpy(npy_data)  # Shouldn't make a copy.

    def __getitem__(self, index):
        img = self.data[index, ...]

        if self.transform is not None:
            img = self.transform(img)

        # Add a bogus label to be compatible with standard image data.
        return img, torch.tensor([0.0])

    def __len__(self):
        return self.data.shape[0]

    def _download(self):
        os.makedirs(self.root, exist_ok=True)

        for tag in ["train", "valid"]:
            npy = os.path.join(self.root, self.NPY_NAME[tag])
            if not os.path.isfile(npy):
                print("Downloading {}...".format(self.NPY_NAME[tag]))
                download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID[tag], npy)


class Preprocess:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_bins = 2 ** self.num_bits

    def __call__(self, img):
        if img.dtype == torch.uint8:
            img = img.float()  # Already in [0,255]
        else:
            img = img * 255.0  # [0,1] -> [0,255]

        if self.num_bits != 8:
            img = torch.floor(img / 2 ** (8 - self.num_bits))  # [0, 255] -> [0, num_bins - 1]

        # Uniform dequantization.
        img = img + torch.rand_like(img)

        # Rescale to (-1., 1.)
        img = -1.0 + img / 128.0

        return img

    def inverse(self, inputs):
        # Rescale from (-1., 1.) to (0., 256.)
        inputs = (inputs + 1.0) * 128.0

        # Discretize the pixel values.
        inputs = torch.floor(inputs)
        # Convert to a float in [0, 1].
        inputs = inputs * (256 / self.num_bins) / 255
        inputs = torch.clamp(inputs, 0, 1)
        return inputs


class RandomHorizontalFlipTensor(object):
    """Random horizontal flip of a CHW image tensor."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        assert img.dim() == 3
        if np.random.rand() < self.p:
            return img.flip(2)  # Flip the width dimension, assuming img shape is CHW.
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class CIFAR10Loader(BaseSimulator):
    def is_image(self):
        return True

    def data_dim(self):
        return (3, 32, 32)

    def parameter_dim(self):
        return None

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0):
        if numpy:
            raise NotImplementedError

        assert limit_samplesize is None
        num_bits = 8
        train_transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(num_bits)])
        test_transform = Preprocess(num_bits)
        return CIFAR10Fast(root=dataset_dir, train=train, download=True, transform=train_transform if train else test_transform)


class ImageNetLoader(BaseSimulator):
    def data_dim(self):
        return (3, 64, 64)

    def is_image(self):
        return True

    def parameter_dim(self):
        return None

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0):
        if numpy:
            raise NotImplementedError

        assert limit_samplesize is None
        num_bits = 8
        return ImageNet64Fast(root=dataset_dir, train=train, download=True, transform=Preprocess(num_bits))


class CelebALoader(BaseSimulator):
    def data_dim(self):
        return (3, 64, 64)

    def is_image(self):
        return True

    def parameter_dim(self):
        return None

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0):
        if numpy:
            raise NotImplementedError

        assert limit_samplesize is None
        num_bits = 8
        train_transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(num_bits)])
        test_transform = Preprocess(num_bits)
        return CelebAHQ64Fast(root=dataset_dir, train=train, download=True, transform=train_transform if train else test_transform)
