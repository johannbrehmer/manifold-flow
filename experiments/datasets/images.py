import os
import logging
import numpy as np
from torchvision import transforms as tvt

from experiments.datasets.utils import Preprocess, RandomHorizontalFlipTensor
from .base import BaseSimulator
from .utils import download_file_from_google_drive, UnlabelledImageDataset

logger = logging.getLogger(__name__)


class BaseImageLoader(BaseSimulator):
    def __init__(self, resolution, n_bits=8, random_horizontal_flips=True, gdrive_file_ids=None):
        self.gdrive_file_ids = gdrive_file_ids
        self.resolution = resolution
        self.n_bits = n_bits
        self.random_horizontal_flips = random_horizontal_flips

    def is_image(self):
        return True

    def data_dim(self):
        return (3, self.resolution, self.resolution)

    def parameter_dim(self):
        return None

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0):
        # Load data as numpy array
        if self.gdrive_file_ids is not None:
            self._download()
        x = np.load("{}/{}.npy".format(dataset_dir, "train" if train else "valid"))

        # Optionally limit sample size
        if limit_samplesize is not None:
            logger.info("Only using %s of %s available samples", limit_samplesize, x.shape[0])
            x = x[:limit_samplesize]

        if numpy:
            # TODO: implement transforms here as well
            return x, None

        # Transforms
        if train and self.random_horizontal_flips:
            transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(self.n_bits)])
        else:
            transform = Preprocess(self.n_bits)

        # Dataset
        dataset = UnlabelledImageDataset(x, transform=transform)
        return dataset

    def _download(self, dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

        for tag in ["train", "valid"]:
            filename = "{}/{}.npy".format(dataset_dir, tag)
            if not os.path.isfile(filename):
                assert self.gdrive_file_ids is not None
                logger.info("Downloading {}.npy".format(tag))
                download_file_from_google_drive(self.gdrive_file_ids[tag], filename)


class ImageNetLoader(BaseImageLoader):
    def __init__(self):
        super().__init__(
            resolution=64,
            n_bits=8,
            random_horizontal_flips=False,
            gdrive_file_ids={"train": "15AMmVSX-LDbP7LqC3R9Ns0RPbDI9301D", "valid": "1Me8EhsSwWbQjQ91vRG1emkIOCgDKK4yC"},
        )


class CelebALoader(BaseImageLoader):
    def __init__(self):
        super().__init__(
            resolution=64,
            n_bits=8,
            random_horizontal_flips=True,
            gdrive_file_ids={"train": "1bcaqMKWzJ-2ca7HCQrUPwN61lfk115TO", "valid": "1WfE64z9FNgOnLliGshUDuCrGBfJSwf-t"},
        )


# class ImageNet64Fast(Dataset):
#     GOOGLE_DRIVE_FILE_ID = {"train": "15AMmVSX-LDbP7LqC3R9Ns0RPbDI9301D", "valid": "1Me8EhsSwWbQjQ91vRG1emkIOCgDKK4yC"}
#
#     NPY_NAME = {"train": "train_64x64.npy", "valid": "valid_64x64.npy"}
#
#     def __init__(self, root, train=True, download=False, transform=None):
#         self.transform = transform
#         self.root = root
#
#         if download:
#             self._download()
#
#         tag = "train" if train else "valid"
#         npy_data = np.load(os.path.join(root, self.NPY_NAME[tag]))
#         self.data = torch.from_numpy(npy_data)  # Shouldn't make a copy.
#
#     def __getitem__(self, index):
#         img = self.data[index, ...]
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         # Add a bogus label to be compatible with standard image data.
#         return img, torch.tensor([0.0])
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def _download(self):
#         os.makedirs(self.root, exist_ok=True)
#
#         for tag in ["train", "valid"]:
#             npy = os.path.join(self.root, self.NPY_NAME[tag])
#             if not os.path.isfile(npy):
#                 logger.info("Downloading {}...".format(self.NPY_NAME[tag]))
#                 download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID[tag], npy)
#
#
# class CelebAHQ64Fast(Dataset):
#     GOOGLE_DRIVE_FILE_ID = {"train": "1bcaqMKWzJ-2ca7HCQrUPwN61lfk115TO", "valid": "1WfE64z9FNgOnLliGshUDuCrGBfJSwf-t"}
#
#     NPY_NAME = {"train": "train.npy", "valid": "valid.npy"}
#
#     def __init__(self, root, train=True, download=False, transform=None):
#         self.transform = transform
#         self.root = root
#
#         if download:
#             self._download()
#
#         tag = "train" if train else "valid"
#         npy_data = np.load(os.path.join(root, self.NPY_NAME[tag]))
#         self.data = torch.from_numpy(npy_data)  # Shouldn't make a copy.
#
#     def __getitem__(self, index):
#         img = self.data[index, ...]
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         # Add a bogus label to be compatible with standard image data.
#         return img, torch.tensor([0.0])
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def _download(self):
#         os.makedirs(self.root, exist_ok=True)
#
#         for tag in ["train", "valid"]:
#             npy = os.path.join(self.root, self.NPY_NAME[tag])
#             if not os.path.isfile(npy):
#                 print("Downloading {}...".format(self.NPY_NAME[tag]))
#                 download_file_from_google_drive(self.GOOGLE_DRIVE_FILE_ID[tag], npy)


# class CIFAR10Fast(datasets.CIFAR10):
#     def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
#         super().__init__(root, train, transform, target_transform, download)
#
#         self.data = self.data.transpose((0, 3, 1, 2))  # HWC -> CHW.
#         self.data = torch.from_numpy(self.data)  # Shouldn't make a copy.
#         assert self.data.dtype == torch.uint8
#
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#
#         # Don't convert to PIL Image, just to convert back later: slow.
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target


# class CIFAR10Loader(BaseSimulator):
#     def is_image(self):
#         return True
#
#     def data_dim(self):
#         return (3, 32, 32)
#
#     def parameter_dim(self):
#         return None
#
#     def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0):
#         if numpy:
#             raise NotImplementedError
#
#         assert limit_samplesize is None
#         num_bits = 8
#         train_transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(num_bits)])
#         test_transform = Preprocess(num_bits)
#         return CIFAR10Fast(root=dataset_dir, train=train, download=True, transform=train_transform if train else test_transform)
