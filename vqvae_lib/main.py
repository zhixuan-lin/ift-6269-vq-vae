import argparse
import os
import os.path as osp
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from vqvae_lib.vqvae import VQVAE, VQVAEBase, create_indices_dataset, PixelCNN
from vqvae_lib.utils import Trainer, save_results
import matplotlib.pyplot as plt
import pathlib



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save the dataset')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'discretized_logistic'], help='Choose the loss to use')
    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    prior_lr = 1e-3
    beta = 0.25
    if args.loss_type == 'mse':
        vq_loss_weight = 1.0
        # lr = 3e-4
        lr = 1e-3
    else:
        vq_loss_weight = 10.0
        lr = 1e-3
    num_embed = 512
    embed_dim = 64

    batch_size = 32
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # epochs = 1
    # prior_epochs = 1
    epochs = 20
    prior_epochs = 20
    n_hidden = 128
    res_hidden = 32



    # TODO: Warning, no validation split here
    # No normalization here since it is done in the code
    transform = transforms.Compose([transforms.ToTensor()])
    # Stupid code here
    data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform).data
    train_data, val_data = data[:40000], data[40000:]
    # test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform).data

    trainloader = DataLoader(train_data.transpose(0, 3, 1, 2).astype(np.float32), batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_data.transpose(0, 3, 1, 2).astype(np.float32), batch_size=batch_size, shuffle=False)
    vqvae_base = VQVAEBase(beta=beta, loss_type=args.loss_type, vq_loss_weight=vq_loss_weight, num_embed=num_embed, embed_dim=embed_dim, n_hidden=n_hidden, res_hidden=res_hidden)
    vqvae_base = vqvae_base.to(device)
    vqvae_trainer = Trainer(vqvae_base, trainloader, valloader, lr, device, epochs, print_every=1, grad_clip=None)
    vqvae_train_log, vqvae_val_log = vqvae_trainer.train()

    prior_train_data = create_indices_dataset(trainloader, vqvae_base, device)
    prior_val_data = create_indices_dataset(valloader, vqvae_base, device)
    prior_trainloader = DataLoader(prior_train_data, batch_size=batch_size, shuffle=True)
    prior_valloader = DataLoader(prior_val_data, batch_size=batch_size, shuffle=False)
    vqvae_prior = PixelCNN(image_shape=(8, 8, 1), channel_ordered=False, n_colors=num_embed, n_layers=8, n_filters=64)
    vqvae_prior = vqvae_prior.to(device)
    prior_trainer = Trainer(vqvae_prior, prior_trainloader, prior_valloader, prior_lr, device, prior_epochs, print_every=1)
    prior_train_log, prior_val_log = prior_trainer.train()


    vqvae_train_loss = vqvae_train_log['loss']
    vqvae_val_loss = vqvae_val_log['loss']

    prior_train_loss = prior_train_log['loss']
    prior_val_loss = prior_val_log['loss']

    # valing simple model saving and loading

    save_path = osp.join(args.result_dir, 'model.pth')
    torch.save(dict(vqvae_base=vqvae_base.state_dict(), vqvae_prior=vqvae_prior.state_dict()), save_path)
    checkpoint = torch.load(save_path, map_location=device)
    vqvae_base.load_state_dict(checkpoint['vqvae_base'])
    vqvae_prior.load_state_dict(checkpoint['vqvae_prior'])



    vqvae = VQVAE(base=vqvae_base, prior=vqvae_prior)
    vqvae.eval()
    samples = vqvae.sample(100)
    # samples = np.random.rand(100, 32, 32, 3)


    # # (100, C, H, W)
    # samples = model.sample(100)

    # assert torch.all((0 <= samples) & (samples <= 255))
    samples = samples.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    # Reconstruction
    images = torch.from_numpy(val_data[:50].transpose(0, 3, 1, 2).astype(np.float32)).to(device)
    recon = vqvae.reconstruct(images)
    recon = recon.cpu().numpy().transpose(0, 2, 3, 1)

    # (100, C, H, W), uint8
    _, H, W, C = recon.shape
    real_recon = np.stack((val_data[:100 // 2], recon), axis=1).astype(np.uint8).reshape(100, H, W, C)
    save_results(samples, real_recon, vqvae_train_loss, vqvae_val_loss, prior_train_loss, prior_val_loss, result_dir=args.result_dir, show_figure=False)



if __name__ == '__main__':
    main()
