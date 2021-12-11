import torch
import numpy as np
import argparse
import os.path as osp
from vqvae_lib.api import train_vanilla_vae
from torchvision.datasets import CIFAR10
from vqvae_lib.utils import SingleTensorDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='data_dir/CIFAR will contain the dataset')
    parser.add_argument('--result_dir', type=str, default='./results/vanilla', help='where to save everything (model/tensorboard/images...)')


    args = parser.parse_args()
    data = CIFAR10(osp.join(args.data_dir), train=True).data
    assert data.shape[0] == 50000
    train_data, val_data = data[:40000], data[40000:]
    to_tensor = lambda x: torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32))
    train_dataset = SingleTensorDataset(to_tensor(train_data))
    val_dataset = SingleTensorDataset(to_tensor(val_data))

    # Adapt to other dataset: change image_size, train_dataset and val_dataset
    train_vanilla_vae(
        image_size=32,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        result_dir=args.result_dir,
        exp_name='vanilla_vae_dl_cifar10',
        loss_type='discretized_logistic',
        epochs=100,
    )
    # train_data, val_data = np.zeros((100, 32, 32, 3)), np.zeros((10000, 32, 32, 3))

if __name__ == '__main__':
    main()
