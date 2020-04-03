import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension, IMG_EXTENSIONS


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


class UnlabelledImageFolder(Dataset):
    """ Image folder dataset. """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = self.find_images(os.path.join(root))

    def __getitem__(self, index):
        path = self.paths[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        # Add a bogus label to be compatible with standard image simulators.
        return image, torch.tensor([0.0])

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def find_images(dir):
        paths = []
        for fname in sorted(os.listdir(dir)):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(dir, fname)
                paths.append(path)
        return paths