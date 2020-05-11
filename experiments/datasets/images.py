import os
import logging
import numpy as np
from torchvision import transforms as tvt

from .utils import Preprocess, RandomHorizontalFlipTensor
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

    def latent_dim(self):
        raise NotImplementedError

    def parameter_dim(self):
        return None

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0, joint_score=False):
        if joint_score:
            raise NotImplementedError("SCANDAL training not implemented for this dataset")

        # Load data as numpy array
        if self.gdrive_file_ids is not None:
            self._download(dataset_dir)
        x = np.load("{}/{}.npy".format(dataset_dir, "train" if train else "test"))

        # Optionally limit sample size
        if limit_samplesize is not None:
            logger.info("Only using %s of %s available samples", limit_samplesize, x.shape[0])
            x = x[:limit_samplesize]

        if numpy:
            # TODO: implement transforms here as well
            logger.warning("Loading image data as numpy array, these data do not have preprocessing applied!")
            return x, None

        # Transforms
        if train and self.random_horizontal_flips:
            transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(self.n_bits)])
        else:
            transform = Preprocess(self.n_bits)

        # Dataset
        dataset = UnlabelledImageDataset(x, transform=transform)
        return dataset

    def sample(self, n, parameters=None):
        raise NotImplementedError

    def sample_ood(self, n, parameters=None):
        raise NotImplementedError

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _download(self, dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

        for tag in ["train", "test"]:
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


class FFHQStyleGAN2DLoader(BaseImageLoader):
    def __init__(self):
        super().__init__(resolution=64, n_bits=8, random_horizontal_flips=True)

    def latent_dim(self):
        return 2
