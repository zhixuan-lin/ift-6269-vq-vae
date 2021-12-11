import torch
import numpy as np
import argparse
import os.path as osp
from vqvae_lib.api import train_gsvae
from vqvae_lib.data import TinyImagenet
from vqvae_lib.utils import SingleTensorDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='data_dir/tiny_imagenet200 will contain the dataset')
    parser.add_argument('--result_dir', type=str, default='./results/gumbel', help='where to save everything (model/tensorboard/images...)')
    args = parser.parse_args()

    train_dataset = TinyImagenet(args.data_dir, TinyImagenet.TRAIN)
    val_dataset = TinyImagenet(args.data_dir, TinyImagenet.VALIDATION)

    # Adapt to other dataset: change image_size, train_dataset and val_dataset
    train_gsvae(
        image_size=64,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        result_dir=args.result_dir,
        exp_name='gsvae_mse_tiny_imagenet',
        loss_type='mse',
        epochs=100,
        prior_epochs=100,
    )
    # train_data, val_data = np.zeros((100, 32, 32, 3)), np.zeros((10000, 32, 32, 3))

if __name__ == '__main__':
    main()
