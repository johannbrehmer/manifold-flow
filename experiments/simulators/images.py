import os
import logging
import numpy as np
import zipfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvt
from torchvision import datasets

from experiments.simulators.base import BaseSimulator
from experiments.utils import download_file_from_google_drive
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

        # Add a bogus label to be compatible with standard image simulators.
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

        return img

    def inverse(self, inputs):
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


# def get_data(dataset_name, num_bits, dataset_root, train=True):
#     if dataset_name == "imagenet-64-fast":
#         root = os.path.join(dataset_root, "imagenet64_fast")
#         c, h, w = (3, 64, 64)
#
#         if train:
#             dataset = ImageNet64Fast(
#                 root=root, train=True, download=True, transform=Preprocess(num_bits)
#             )
#
#             # num_train = len(train_dataset)
#             # valid_size = int(np.floor(num_train * valid_frac))
#             # train_size = num_train - valid_size
#             # train_dataset, valid_dataset = random_split(train_dataset,
#             #                                             (train_size, valid_size))
#         else:
#             dataset = ImageNet64Fast(
#                 root=root, train=False, download=True, transform=Preprocess(num_bits)
#             )
#
#     elif dataset_name == "cifar-10-fast" or dataset_name == "cifar-10":
#         root = os.path.join(dataset_root, "cifar-10")
#         c, h, w = (3, 32, 32)
#
#         if dataset_name == "cifar-10-fast":
#             dataset_class = CIFAR10Fast
#             train_transform = tvt.Compose(
#                 [RandomHorizontalFlipTensor(), Preprocess(num_bits)]
#             )
#             test_transform = Preprocess(num_bits)
#         else:
#             dataset_class = datasets.CIFAR10
#             train_transform = tvt.Compose(
#                 [tvt.RandomHorizontalFlip(), tvt.ToTensor(), Preprocess(num_bits)]
#             )
#             test_transform = tvt.Compose([tvt.ToTensor(), Preprocess(num_bits)])
#
#         if train:
#             dataset = dataset_class(
#                 root=root, train=True, download=True, transform=train_transform
#             )
#
#             # valid_dataset = dataset_class(
#             #     root=root,
#             #     train=True,
#             #     transform=test_transform # Note different transform.
#             # )
#
#             # num_train = len(train_dataset)
#             # indices = torch.randperm(num_train).tolist()
#             # valid_size = int(np.floor(valid_frac * num_train))
#             # train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
#
#             # train_dataset = Subset(train_dataset, train_idx)
#             # valid_dataset = Subset(valid_dataset, valid_idx)
#         else:
#             dataset = dataset_class(
#                 root=root, train=False, download=True, transform=test_transform
#             )
#
#     elif dataset_name == "imagenet-32" or dataset_name == "imagenet-64":
#         if dataset_name == "imagenet-32":
#             root = os.path.join(dataset_root, "imagenet32")
#             c, h, w = (3, 32, 32)
#             dataset_class = ImageNet32
#         else:
#             root = os.path.join(dataset_root, "imagenet64")
#             c, h, w = (3, 64, 64)
#             dataset_class = ImageNet64
#
#         if train:
#             dataset = dataset_class(
#                 root=root,
#                 train=True,
#                 download=True,
#                 transform=tvt.Compose([tvt.ToTensor(), Preprocess(num_bits)]),
#             )
#
#             # num_train = len(train_dataset)
#             # valid_size = int(np.floor(num_train * valid_frac))
#             # train_size = num_train - valid_size
#             # train_dataset, valid_dataset = random_split(train_dataset,
#             #                                             (train_size, valid_size))
#         else:
#             dataset = dataset_class(
#                 root=root,
#                 train=False,
#                 download=True,
#                 transform=tvt.Compose([tvt.ToTensor(), Preprocess(num_bits)]),
#             )
#     # elif dataset_name == "celeba-hq-64-fast":
#     #     root = os.path.join(dataset_root, "celeba_hq_64_fast")
#     #     c, h, w = (3, 64, 64)
#     #
#     #     train_transform = tvt.Compose(
#     #         [RandomHorizontalFlipTensor(), Preprocess(num_bits)]
#     #     )
#     #     test_transform = Preprocess(num_bits)
#     #
#     #     if train:
#     #         dataset = CelebAHQ64Fast(
#     #             root=root, train=True, download=True, transform=train_transform
#     #         )
#     #
#     #         # valid_dataset = data.CelebAHQ64Fast(
#     #         #     root=root,
#     #         #     train=True,
#     #         #     transform=test_transform # Note different transform.
#     #         # )
#     #
#     #         # num_train = len(train_dataset)
#     #         # indices = torch.randperm(num_train).tolist()
#     #         # valid_size = int(np.floor(valid_frac * num_train))
#     #         # train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
#     #         #
#     #         # train_dataset = Subset(train_dataset, train_idx)
#     #         # valid_dataset = Subset(valid_dataset, valid_idx)
#     #     else:
#     #         dataset = CelebAHQ64Fast(
#     #             root=root, train=False, download=True, transform=test_transform
#     #         )
#
#     else:
#         raise RuntimeError("Unknown dataset {}".format(dataset_name))
#
#     return dataset, (c, h, w)


class CIFAR10Loader(BaseSimulator):
    def is_image(self):
        return True

    def data_dim(self):
        return (3, 32, 32)

    def parameter_dim(self):
        return None

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None):
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

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None):
        if numpy:
            raise NotImplementedError

        assert limit_samplesize is None
        num_bits = 8
        return ImageNet64Fast(root=dataset_dir, train=train, download=True, transform=Preprocess(num_bits))
