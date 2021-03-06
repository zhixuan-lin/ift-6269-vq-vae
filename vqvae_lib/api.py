
import math
import argparse
import os
import os.path as osp
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from vqvae_lib.vqvae import VQVAE, VQVAEBase, create_indices_dataset, VQVAEPrior
from vqvae_lib.baselines import GumbelSoftmaxVAE, GumbelSoftmaxVAEBase, GumbelSoftmaxVAEPrior
from vqvae_lib.utils import Trainer, save_results, attach_run_id, CSVWriter
import matplotlib.pyplot as plt
import pathlib
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def evaluate_loglikelihood(discrete_type, train_dataset, val_dataset, result_dir, exp_name_with_id, image_size, device, loss_type, num_embed=512, embed_dim=64, n_hidden=128, res_hidden=32, tau_start=1.0, batch_size=32):
    assert osp.exists(result_dir)
    if device =='auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_path = osp.join(result_dir, exp_name_with_id, 'model.pth')
    checkpoint = torch.load(save_path, map_location=device)
    latent_size = image_size // 4
    if discrete_type == 'vq':
        vqvae_base = VQVAEBase(loss_type=loss_type, num_embed=num_embed, embed_dim=embed_dim, n_hidden=n_hidden, res_hidden=res_hidden).to(device)
        vqvae_prior = VQVAEPrior(image_shape=(latent_size, latent_size, 1), channel_ordered=False, n_colors=num_embed, n_layers=8, n_filters=64).to(device)
        vqvae_base.load_state_dict(checkpoint['vqvae_base'])
        vqvae_prior.load_state_dict(checkpoint['vqvae_prior'])
        model = VQVAE(vqvae_base, vqvae_prior)
    elif discrete_type =='gs':
        gsvae_base = GumbelSoftmaxVAEBase(loss_type=loss_type, num_embed=num_embed, embed_dim=embed_dim, n_hidden=n_hidden, res_hidden=res_hidden, tau=tau_start).to(device)
        gsvae_prior = GumbelSoftmaxVAEPrior(image_shape=(latent_size, latent_size, 1), channel_ordered=False, n_colors=num_embed, n_layers=8, n_filters=64).to(device)
        gsvae_base.load_state_dict(checkpoint['gsvae_base'])
        gsvae_prior.load_state_dict(checkpoint['gsvae_prior'])
        model = GumbelSoftmaxVAE(gsvae_base, gsvae_prior)
    else:
        raise ValueError(f'Invalid discrete type {discrete_type}')

    train_loglike_total = 0.0
    train_count = 0
    print('Evaluating train log likelihood..')
    for data in trainloader:
        data = data.to(device)
        B = data.size(0)
        loglike = model.logprob(data)
        assert loglike.size() == (B,)
        train_loglike_total += loglike.sum().item()
        train_count += B

    val_loglike_total = 0.0
    val_count = 0
    print('Evaluating val log likelihood..')
    for data in valloader:
        data = data.to(device)
        B = data.size(0)
        loglike = model.logprob(data)
        assert loglike.size() == (B,)
        val_loglike_total += loglike.sum().item()
        val_count += B

    train_loglike = train_loglike_total / train_count
    val_loglike = val_loglike_total / val_count
    print('Train log')
    logpath = osp.join(result_dir, 'loglikelihood.txt')
    with open(logpath, 'w') as f:
        print('Train loglikelihood: ', train_loglike, file=f)
        print('val loglikelihood: ', val_loglike, file=f)
    print('Train loglikelihood: ', train_loglike)
    print('val loglikelihood: ', val_loglike)


def train_vqvae(
    image_size=32,
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    result_dir='./results',
    exp_name='run',
    loss_type='mse',
    device='auto',
    lr=3e-4,
    prior_lr=1e-3,
    beta=0.25,
    vq_loss_weight=1.0,
    num_embed=512,
    embed_dim=64,
    batch_size=32,
    epochs=20,
    prior_epochs=20,
    n_hidden=128,
    res_hidden=32
):
    assert train_dataset is not None and val_dataset is not None


    exp_name = attach_run_id(result_dir, exp_name)
    if device =='auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    summary_writer = SummaryWriter(log_dir=osp.join(result_dir, exp_name))
    csv_writer = CSVWriter(log_dir=osp.join(result_dir, exp_name))
    prior_summary_writer = SummaryWriter(log_dir=osp.join(result_dir, exp_name, 'prior'))
    prior_csv_writer = CSVWriter(log_dir=osp.join(result_dir, exp_name, 'prior'))




    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    vqvae_base = VQVAEBase(beta=beta, loss_type=loss_type, vq_loss_weight=vq_loss_weight, num_embed=num_embed, embed_dim=embed_dim, n_hidden=n_hidden, res_hidden=res_hidden)
    vqvae_base = vqvae_base.to(device)
    vqvae_trainer = Trainer(vqvae_base, trainloader, valloader, lr, device, epochs, print_every=1, grad_clip=None, summary_writer=summary_writer, csv_writer=csv_writer)
    vqvae_train_log, vqvae_val_log = vqvae_trainer.train()

    prior_train_data = create_indices_dataset(trainloader, vqvae_base, device)
    prior_val_data = create_indices_dataset(valloader, vqvae_base, device)
    prior_trainloader = DataLoader(prior_train_data, batch_size=batch_size, shuffle=True)
    prior_valloader = DataLoader(prior_val_data, batch_size=batch_size, shuffle=False)
    latent_size = image_size // 4
    vqvae_prior = VQVAEPrior(image_shape=(latent_size, latent_size, 1), channel_ordered=False, n_colors=num_embed, n_layers=8, n_filters=64)
    vqvae_prior = vqvae_prior.to(device)
    prior_trainer = Trainer(vqvae_prior, prior_trainloader, prior_valloader, prior_lr, device, prior_epochs, print_every=1, summary_writer=prior_summary_writer, csv_writer=prior_csv_writer)
    prior_train_log, prior_val_log = prior_trainer.train()


    vqvae_train_loss = vqvae_train_log['loss']
    vqvae_val_loss = vqvae_val_log['loss']

    prior_train_loss = prior_train_log['loss']
    prior_val_loss = prior_val_log['loss']

    # valing simple model saving and loading

    save_path = osp.join(result_dir, exp_name, 'model.pth')
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
    # images = torch.from_numpy(val_data[:50].transpose(0, 3, 1, 2).astype(np.float32)).to(device)
    images = torch.stack([val_dataset[i] for i in range(50)], dim=0).to(device)
    recon = vqvae.reconstruct(images)

    # (100, C, H, W), uint8
    _, C, H, W = recon.shape
    real_recon = torch.stack((images, recon), axis=1).view(100, C, H, W)
    real_recon = real_recon.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    save_results(samples, real_recon, vqvae_train_loss, vqvae_val_loss, prior_train_loss, prior_val_loss, result_dir=osp.join(result_dir, exp_name), show_figure=False)


def train_vanilla_vae(
    image_size=32,
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    result_dir='./results/vanilla',
    exp_name='run',
    loss_type='mse',
    device='auto',
    lr=3e-4,
    beta=1.0,
    num_embed=512,
    embed_dim=64,
    batch_size=32,
    epochs=20,
    n_hidden=128,
    res_hidden=32
):
    assert train_dataset is not None and val_dataset is not None

    exp_name = attach_run_id(result_dir, exp_name)
    if device =='auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    summary_writer = SummaryWriter(log_dir=osp.join(result_dir, exp_name))
    csv_writer = CSVWriter(log_dir=osp.join(result_dir, exp_name))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    vanilla_vae = VanillaVAE(beta=beta, loss_type=loss_type, num_embed=num_embed, embed_dim=embed_dim, n_hidden=n_hidden, res_hidden=res_hidden, image_width=image_size)
    vanilla_vae = vanilla_vae.to(device)
    vanilla_vae_trainer = Trainer(vanilla_vae, trainloader, valloader, lr, device, epochs, print_every=1, grad_clip=None, summary_writer=summary_writer, csv_writer=csv_writer)
    vanilla_vae_train_log, vanilla_vae_val_log = vanilla_vae_trainer.train()

    vanilla_vae_train_loss = vanilla_vae_train_log['loss']
    vanilla_vae_val_loss = vanilla_vae_val_log['loss']

    # valing simple model saving and loading
    save_path = osp.join(result_dir, exp_name, 'model.pth')
    torch.save(dict(vanilla_vae=vanilla_vae.state_dict()), save_path)
    checkpoint = torch.load(save_path, map_location=device)
    vanilla_vae.load_state_dict(checkpoint['vanilla_vae'])

    vanilla_vae.eval()
    samples = vanilla_vae.sample(100)

    # assert torch.all((0 <= samples) & (samples <= 255))
    samples = samples.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    # Reconstruction
    images = torch.stack([val_dataset[i] for i in range(50)], dim=0).to(device)
    recon = vanilla_vae.reconstruct(images)

    # (100, C, H, W), uint8
    _, C, H, W = recon.shape
    real_recon = torch.stack((images, recon), axis=1).view(100, C, H, W)
    real_recon = real_recon.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    save_results(samples, real_recon, vanilla_vae_train_loss, vanilla_vae_val_loss, prior_train_loss=None, prior_test_loss=None, result_dir=osp.join(result_dir, exp_name), show_figure=False)


def train_gsvae(
    image_size=32,
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    result_dir='./results/gumbel',
    exp_name='run',
    loss_type='mse',
    device='auto',
    lr=3e-4,
    prior_lr=1e-3,
    beta=0.25,
    vq_loss_weight=1.0,
    num_embed=512,
    embed_dim=64,
    batch_size=32,
    epochs=20,
    prior_epochs=20,
    n_hidden=128,
    res_hidden=32,
    tau_start=1.0
):
    assert train_dataset is not None and val_dataset is not None

    class AdjustTau:
        def __init__(self, tau_start, model, max_epochs):
            self.tau_start = tau_start
            self.model = model
            self.max_epochs = max_epochs
        
        def __call__(self, epoch):
            """Decay the learning rate based on schedule"""
            # cosine schedule
            tau = self.tau_start
            if epoch > self.max_epochs:
                tau = 1 / 16
            else:
                tau *= 0.5 * (1. + math.cos(math.pi * epoch / self.max_epochs))
                tau = max(tau, 1 / 16)
            self.model.tau = tau

    exp_name = attach_run_id(result_dir, exp_name)
    if device =='auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    summary_writer = SummaryWriter(log_dir=osp.join(result_dir, exp_name))
    csv_writer = CSVWriter(log_dir=osp.join(result_dir, exp_name))
    prior_summary_writer = SummaryWriter(log_dir=osp.join(result_dir, exp_name, 'prior'))
    prior_csv_writer = CSVWriter(log_dir=osp.join(result_dir, exp_name, 'prior'))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    gsvae_base = GumbelSoftmaxVAEBase(beta=beta, loss_type=loss_type, vq_loss_weight=vq_loss_weight, num_embed=num_embed, embed_dim=embed_dim, n_hidden=n_hidden, res_hidden=res_hidden, tau=tau_start)
    gsvae_base = gsvae_base.to(device)
    # adjust_tau = AdjustTau(tau_start, gsvae_base, max_epochs=epochs)
    gsvae_trainer = Trainer(gsvae_base, trainloader, valloader, lr, device, epochs, print_every=1, grad_clip=None, summary_writer=summary_writer, csv_writer=csv_writer)
    gsvae_train_log, gsvae_val_log = gsvae_trainer.train()

    prior_train_data = create_indices_dataset(trainloader, gsvae_base, device)
    prior_val_data = create_indices_dataset(valloader, gsvae_base, device)
    prior_trainloader = DataLoader(prior_train_data, batch_size=batch_size, shuffle=True)
    prior_valloader = DataLoader(prior_val_data, batch_size=batch_size, shuffle=False)
    latent_size = image_size // 4
    gsvae_prior = GumbelSoftmaxVAEPrior(image_shape=(latent_size, latent_size, 1), channel_ordered=False, n_colors=num_embed, n_layers=8, n_filters=64)
    gsvae_prior = gsvae_prior.to(device)
    prior_trainer = Trainer(gsvae_prior, prior_trainloader, prior_valloader, prior_lr, device, prior_epochs, print_every=1, summary_writer=prior_summary_writer, csv_writer=prior_csv_writer)
    prior_train_log, prior_val_log = prior_trainer.train()


    gsvae_train_loss = gsvae_train_log['loss']
    gsvae_val_loss = gsvae_val_log['loss']

    prior_train_loss = prior_train_log['loss']
    prior_val_loss = prior_val_log['loss']

    # valing simple model saving and loading

    save_path = osp.join(result_dir, exp_name, 'model.pth')
    torch.save(dict(gsvae_base=gsvae_base.state_dict(), gsvae_prior=gsvae_prior.state_dict()), save_path)
    checkpoint = torch.load(save_path, map_location=device)
    gsvae_base.load_state_dict(checkpoint['gsvae_base'])
    gsvae_prior.load_state_dict(checkpoint['gsvae_prior'])

    gsvae = GumbelSoftmaxVAE(base=gsvae_base, prior=gsvae_prior)
    gsvae.eval()
    samples = gsvae.sample(100)
    # samples = np.random.rand(100, 32, 32, 3)


    # # (100, C, H, W)
    # samples = model.sample(100)

    # assert torch.all((0 <= samples) & (samples <= 255))
    samples = samples.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    # Reconstruction
    # images = torch.from_numpy(val_data[:50].transpose(0, 3, 1, 2).astype(np.float32)).to(device)
    images = torch.stack([val_dataset[i] for i in range(50)], dim=0).to(device)
    recon = gsvae.reconstruct(images)

    # (100, C, H, W), uint8
    _, C, H, W = recon.shape
    real_recon = torch.stack((images, recon), axis=1).view(100, C, H, W)
    real_recon = real_recon.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    save_results(samples, real_recon, gsvae_train_loss, gsvae_val_loss, prior_train_loss, prior_val_loss, result_dir=osp.join(result_dir, exp_name), show_figure=False)
