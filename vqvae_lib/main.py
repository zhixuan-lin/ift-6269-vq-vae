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
from vqvae_lib.utils import Trainer, save_results
import matplotlib.pyplot as plt
import pathlib



def main():
    lr = 1e-3
    prior_lr = 1e-3
    batch_size = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    epochs = 20
    prior_epochs = 100
    # epochs = 20
    # prior_epochs = 200

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save the dataset')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'discretized_logistic'], help='Choose the loss to use')
    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)



    # TODO: Warning, no validation split here
    # No normalization here since it is done in the code
    transform = transforms.Compose([transforms.ToTensor()])
    # Stupid code here
    train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform).data
    test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform).data

    trainloader = DataLoader(train_data.transpose(0, 3, 1, 2).astype(np.float32), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data.transpose(0, 3, 1, 2).astype(np.float32), batch_size=batch_size, shuffle=False)
    vqvae_base = VQVAEBase(beta=1., loss_type=args.loss_type)
    vqvae_base = vqvae_base.to(device)
    vqvae_trainer = Trainer(vqvae_base, trainloader, testloader, lr, device, epochs, print_every=1, grad_clip=1.)
    vqvae_train_log, vqvae_test_log = vqvae_trainer.train()

    prior_train_data = create_indices_dataset(trainloader, vqvae_base, device)
    prior_test_data = create_indices_dataset(testloader, vqvae_base, device)
    prior_trainloader = DataLoader(prior_train_data, batch_size=batch_size, shuffle=True)
    prior_testloader = DataLoader(prior_test_data, batch_size=batch_size, shuffle=False)
    vqvae_prior = VQVAEPrior(image_shape=(8, 8, 1), channel_ordered=False, n_colors=128, n_layers=8, n_filters=64)
    vqvae_prior = vqvae_prior.to(device)
    prior_trainer = Trainer(vqvae_prior, prior_trainloader, prior_testloader, prior_lr, device, prior_epochs, print_every=1)
    prior_train_log, prior_test_log = prior_trainer.train()


    vqvae_train_loss = vqvae_train_log['loss']
    vqvae_test_loss = vqvae_test_log['loss']

    prior_train_loss = prior_train_log['loss']
    prior_test_loss = prior_test_log['loss']

    # Testing simple model saving and loading

    save_path = osp.join(args.result_dir, 'model.pth')
    torch.save(dict(vqvae_base=vqvae_base.state_dict(), vqvae_prior=vqvae_prior.state_dict()), save_path)
    checkpoint = torch.load(save_path, map_location=device)
    vqvae_base.load_state_dict(checkpoint['vqvae_base'])
    vqvae_prior.load_state_dict(checkpoint['vqvae_prior'])



    vqvae = VQVAE(base=vqvae_base, prior=vqvae_prior)
    samples = vqvae.sample(100)
    # samples = np.random.rand(100, 32, 32, 3)


    # # (100, C, H, W)
    # samples = model.sample(100)

    # assert torch.all((0 <= samples) & (samples <= 255))
    samples = samples.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    # Reconstruction
    images = torch.from_numpy(test_data[:50].transpose(0, 3, 1, 2).astype(np.float32)).to(device)
    recon = vqvae.reconstruct(images)
    recon = recon.cpu().numpy().transpose(0, 2, 3, 1)

    # (100, C, H, W), uint8
    real_recon = np.concatenate((test_data[:50], recon), axis=0).astype(np.uint8)
    save_results(samples, real_recon, vqvae_train_loss, vqvae_test_loss, prior_train_loss, prior_test_loss, result_dir=args.result_dir, show_figure=False)



if __name__ == '__main__':
    main()
