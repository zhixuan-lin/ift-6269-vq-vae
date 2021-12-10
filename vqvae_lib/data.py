from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import ImageFolder
from pathlib import Path
from tqdm import tqdm
import torchvision
import torch
import urllib.request
import ssl
import sys
import zipfile

def download_url(url: str, folder: Path):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    filename = filename if filename[0] == '?' else filename.split('?')[0]
    path = folder / filename

    if path.exists():
        print(f'Using existing file {filename}', file=sys.stderr)
        return path

    print(f'Downloading {url}', file=sys.stderr)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path

class TinyImagenet(Dataset):
    """Wrapper around an ImageFolder with logic to download tiny imagenet

    """
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'

    _DATASET_TYPES = [
        TRAIN,
        VALIDATION,
        TEST
    ]

    def __init__(self, data_dir: str, dataset_type: str) -> None:
        super().__init__()

        data_dir = Path(data_dir)
        downloaded_dir = self._download(data_dir)

        assert dataset_type in TinyImagenet._DATASET_TYPES

        transform = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Lambda(lambda x: x.to(dtype=torch.float32))
        ])

        data = ImageFolder(
            downloaded_dir / dataset_type,
            transform=transform
        )

        data = torch.stack([img[0] for img in data])
        self.data = TensorDataset(data)

    def _download(self, data_dir: Path) -> None:
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

        filename = data_dir / url.rpartition('/')[2]
        if self._is_already_downloaded(filename):
            return filename.parent / filename.name.split('.')[0]

        data_dir.mkdir(exist_ok=True)
        zip_file_path = Path(download_url(url, data_dir))

        zip_file_dir = zip_file_path.parent / zip_file_path.name.split('.')[0]
        zip_file_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(zip_file_dir)

        return zip_file_dir

    def _is_already_downloaded(self, data_dir: Path) -> bool:
        actual_dir = data_dir.parent / data_dir.name.split('.')[0]
        if not actual_dir.exists():
            return False

        return all(map(
            lambda x: (actual_dir / x).exists(),
            TinyImagenet._DATASET_TYPES
        ))

    def __getitem__(self, idx):
        return self.data[idx][0]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train = TinyImagenet('data_dir', TinyImagenet.TRAIN)
    val = TinyImagenet('data_dir', TinyImagenet.VALIDATION)
    test = TinyImagenet('data_dir', TinyImagenet.TEST)
