import argparse
import torch
import numpy as np
import os.path as osp
from vqvae_lib.api import evaluate_loglikelihood
from torchvision.datasets import CIFAR10
from vqvae_lib.utils import SingleTensorDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='data_dir/CIFAR will contain the dataset')
    parser.add_argument('--result_dir', type=str, default='./results', help='where to save everything (model/tensorboard/images...)')
    parser.add_argument('--exp_name', type=str, default='./results', help='experiement name with run id')


    args = parser.parse_args()
    data = CIFAR10(osp.join(args.data_dir), train=True).data
    assert data.shape[0] == 50000
    train_data, val_data = data[:40000], data[40000:]
    to_tensor = lambda x: torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32))
    train_dataset = SingleTensorDataset(to_tensor(train_data))
    val_dataset = SingleTensorDataset(to_tensor(val_data))

    # Adapt to other dataset: change image_size, train_dataset and val_dataset
    # train_data, val_data = np.zeros((100, 32, 32, 3)), np.zeros((10000, 32, 32, 3))
    evaluate_loglikelihood(
        discrete_type='gs',
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        result_dir=args.result_dir,
        exp_name_with_id=args.exp_name,
        image_size=32,
        device='auto',
        loss_type='discretized_logistic'
    )

if __name__ == '__main__':
    main()
