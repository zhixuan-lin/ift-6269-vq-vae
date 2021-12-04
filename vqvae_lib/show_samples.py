import argparse
import os
import os.path as osp
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from vqvae_lib.vqvae import VQVAE, VQVAEBase, VQVAEPrior, create_indices_dataset
from vqvae_lib.utils import Trainer, save_results, savefig, show_samples
import matplotlib.pyplot as plt
import pathlib


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save the dataset')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'discretized_logistic'], help='Choose the loss to use')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to show')
    parser.add_argument('--nrow', type=int, default=10, help='Samples per row')
    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)



    # TODO: Warning, no validation split here
    # No normalization here since it is done in the code
    transform = transforms.Compose([transforms.ToTensor()])
    # Stupid code here
    # train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform).data
    val_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform).data[40000:]

    vqvae_base = VQVAEBase(loss_type=args.loss_type)
    vqvae_base = vqvae_base.to(device)
    vqvae_prior = VQVAEPrior(image_shape=(8, 8, 1), channel_ordered=False, n_colors=vqvae_base.num_embed, n_layers=8, n_filters=64)
    vqvae_prior = vqvae_prior.to(device)



    save_path = osp.join(args.result_dir, 'model.pth')
    # torch.save(dict(vqvae_base=vqvae_base.state_dict(), vqvae_prior=vqvae_prior.state_dict()), save_path)
    checkpoint = torch.load(save_path, map_location=device)
    vqvae_base.load_state_dict(checkpoint['vqvae_base'])
    vqvae_prior.load_state_dict(checkpoint['vqvae_prior'])



    vqvae = VQVAE(base=vqvae_base, prior=vqvae_prior)
    vqvae.eval()
    samples = vqvae.sample(args.num_samples)
    # samples = np.random.rand(100, 32, 32, 3)


    # # (100, C, H, W)
    # samples = model.sample(100)

    # assert torch.all((0 <= samples) & (samples <= 255))
    samples = samples.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    # Reconstruction
    images = torch.from_numpy(val_data[:args.num_samples // 2].transpose(0, 3, 1, 2).astype(np.float32)).to(device)
    recon = vqvae.reconstruct(images)
    recon = recon.cpu().numpy().transpose(0, 2, 3, 1)

    # Stack and reshape
    # (50, 2, H, W, C) -> (100, H, W, C)
    # real_recon = np.concatenate((val_data[:args.num_samples // 2], recon), axis=0).astype(np.uint8)
    _, H, W, C = recon.shape
    real_recon = np.stack((val_data[:args.num_samples // 2], recon), axis=1).astype(np.uint8).reshape(args.num_samples, H, W, C)
    # save_results(samples, real_recon, vqvae_train_loss, vqvae_val_loss, prior_train_loss, prior_val_loss, result_dir=args.result_dir, show_figure=False)
    # savefig(fname)
    show_samples(real_recon, title='Reconstructions', fname=osp.join(args.result_dir, 'reconstructions.png'), nrow=args.nrow)
    show_samples(samples, title='Samples', fname=osp.join(args.result_dir, 'samples.png'), nrow=args.nrow)



if __name__ == '__main__':
    main()
