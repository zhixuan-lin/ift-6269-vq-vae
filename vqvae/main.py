import argparse
import os
import os.path as osp
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from vqvae import VQVAE, VQVAEBase, VQVAEPrior, create_indices_dataset
import matplotlib.pyplot as plt

class ArrayDict(dict):
  def append(self, x):
    if self:
        assert self.keys() == x.keys()
        for k in x:
            self[k] = np.append(self[k], x[k])
    else:
        for k in x:
            self[k] =  np.array(x[k])
  
  def extend(self, x):
    if self:
        assert self.keys() == x.keys()
        for k in x:
            try:
                self[k] = np.concatenate([self[k], x[k]])
            except:
                import ipdb; ipdb.set_trace()
    else:
        for k in x:
            self[k] = np.array(x[k])



class Trainer:
    def __init__(
        self,
        model,
        trainloader,
        valloader,
        learning_rate,
        device,
        max_epochs,
        print_every=1,
        grad_clip=None
    ):
      self.model = model
      self.trainloader = trainloader
      self.valloader = trainloader
      self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      self.device = device
      self.max_epochs = max_epochs
      self.print_every = print_every
      self.grad_clip = grad_clip



    def train(self):
        test_log = ArrayDict()
        train_log = ArrayDict()
        test_log.append(self.validate())
        print('Initial loss:', test_log['loss'])
        for epoch in range(self.max_epochs):
 
            train_log.extend(self.train_one_epoch())
            test_log.append(self.validate())
            if (epoch + 1) % self.print_every == 0:
                string = 'epoch: {}'.format(epoch + 1)
                string += ', loss: {}'.format(test_log['loss'][-1])
                for k in sorted(test_log):
                    if k == 'loss':
                        continue
                    string += ', {}: {:.4f}'.format(k, test_log[k][-1])
                print(string)
                # print('Epoch: {}, Loss: {}, -Elbo: {}, KL: {}'.format(epoch + 1, test_log['loss'][-1], test_log['neg_elbo'][-1], test_log['kl'][-1]))

        return train_log, test_log

    def train_one_epoch(self):
        logs = ArrayDict()
        for iteration, data in enumerate(self.trainloader):
            self.model.train()
            data = data.to(self.device)
            # with torch.autograd.detect_anomaly():
            loss, log = self.model(data)
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            # Mean within batch
            log = {k: v.mean().item() for (k, v) in log.items()}
            logs.append(log)
        return logs


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        logs = ArrayDict()
        for data in self.valloader:
            # Assume in batch, (B,)
            data = data.to(self.device)
            loss, log = self.model(data)
            log = {k: log[k].cpu().numpy() for k in log}
            logs.extend(log)
        # Mean over all data
        assert all([v.ndim == 1 for v in logs.values()])
        logs = {k: logs[k].mean() for k in logs}
        return logs
    
# from  https://github.com/rll/deepul/blob/master/deepul/utils.py
def savefig(fname, show_figure=True):
    os.makedirs(osp.dirname(fname), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()

# from  https://github.com/rll/deepul/blob/master/deepul/utils.py
def show_samples(samples, fname=None, nrow=10, title='Samples', show_figure=False):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        savefig(fname, show_figure)
    else:
        plt.show()

# from  https://github.com/rll/deepul/blob/master/deepul/utils.py
def save_training_plot(train_losses, test_losses, title, fname, show_figure=False):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    savefig(fname, show_figure)

# Show results, from https://github.com/rll/deepul/blob/master/deepul/hw3_helper.py
def save_results(samples, real_recon, vqvae_train_loss, vqvae_test_loss, prior_train_loss, prior_test_loss, show_figure=False):
    samples, real_recon = samples.astype('float32'), real_recon.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_loss[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {prior_test_loss[-1]:.4f}')
    save_training_plot(vqvae_train_loss, vqvae_test_loss,'VQ-VAE Train Plot',
                       'results/vqvae_train_plot.png')
    save_training_plot(prior_train_loss, prior_test_loss,'PixelCNN Prior Train Plot',
                       'results/prior_train_plot.png')
    show_samples(samples, title='Samples',
                 fname='results/samples.png')
    show_samples(real_recon, title='Reconstructions',
                 fname='results/reconstructions.png')


def main():
    lr = 1e-3
    prior_lr = 1e-3
    batch_size = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    epochs = 20
    prior_epochs = 200
    epochs = 2
    prior_epochs = 2



    # TODO: Warning, no validation split here
    # No normalization here since it is done in the code
    transform = transforms.Compose([transforms.ToTensor()])
    # Stupid code here
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform).data[:256]
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform).data[:256]

    trainloader = DataLoader(train_data.transpose(0, 3, 1, 2).astype(np.float32), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data.transpose(0, 3, 1, 2).astype(np.float32), batch_size=batch_size, shuffle=False)
    vqvae_base = VQVAEBase(beta=1.)
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
    save_results(samples, real_recon, vqvae_train_loss, vqvae_test_loss, prior_train_loss, prior_test_loss, show_figure=False)



if __name__ == '__main__':
    main()
