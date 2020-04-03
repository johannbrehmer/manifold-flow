import os
import numpy as np
import torch
import requests

from torch.utils.data import Dataset


def download_file(url, dest):
    CHUNK_SIZE = 8192

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive new chunks.
                f.write(chunk)


def download_file_from_google_drive(id, dest):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest)


class NumpyDataset(Dataset):
    """ Dataset for numpy arrays with explicit memmap support """

    def __init__(self, *arrays, **kwargs):

        self.dtype = kwargs.get("dtype", torch.float)
        self.memmap = []
        self.data = []
        self.n = None

        memmap_threshold = kwargs.get("memmap_threshold", None)

        for array in arrays:
            if isinstance(array, str):
                array = self._load_array_from_file(array, memmap_threshold)

            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n

    @staticmethod
    def _load_array_from_file(filename, memmap_threshold_gb=None):
        filesize_gb = os.stat(filename).st_size / 1.0 * 1024 ** 3
        if memmap_threshold_gb is None or filesize_gb <= memmap_threshold_gb:
            data = np.load(filename)
        else:
            data = np.load(filename, mmap_mode="c")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return data


class UnlabelledImageDataset(Dataset):
    def __init__(self, array, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(array)

    def __getitem__(self, index):
        img = self.data[index, ...]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor([0.0])

    def __len__(self):
        return self.data.shape[0]


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
