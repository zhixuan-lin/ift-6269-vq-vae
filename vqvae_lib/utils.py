import os
import os.path as osp
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pathlib


class SingleTensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    def __getitem__(self, index):
        return self.tensor[index]
    def __len__(self):
        return self.tensor.size(0)


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
        grad_clip=None,
        summary_writer=None,
        log_every_n_steps=5,
    ):
      self.model = model
      self.trainloader = trainloader
      self.valloader = trainloader
      self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      self.device = device
      self.max_epochs = max_epochs
      self.print_every = print_every
      self.grad_clip = grad_clip
      self.global_step = 0
      self.summary_writer = summary_writer
      self.log_every_n_steps = log_every_n_steps

    def train(self):
        val_log = ArrayDict()
        train_log = ArrayDict()
        val_log.append(self.validate())
        print('Initial loss:', val_log['loss'])
        for epoch in range(self.max_epochs):
 
            train_log.extend(self.train_one_epoch())
            val_log.append(self.validate())
            if (epoch + 1) % self.print_every == 0:
                string = 'Train: epoch: {}'.format(epoch + 1)
                string += ', iter: {}'.format((epoch + 1) * len(self.trainloader))
                string += ', loss: {}'.format(np.mean(train_log['loss'][-100:]))
                for k in sorted(train_log):
                    if k == 'loss':
                        continue
                    string += ', {}: {:.4f}'.format(k, np.mean(train_log[k][-100:]))
                print(string)

                string = 'Val: epoch: {}'.format(epoch + 1)
                string += ', iter: {}'.format((epoch + 1) * len(self.trainloader))
                string += ', loss: {}'.format(val_log['loss'][-1])
                for k in sorted(val_log):
                    if k == 'loss':
                        continue
                    string += ', {}: {:.4f}'.format(k, val_log[k][-1])
                print(string)
                # print('Epoch: {}, Loss: {}, -Elbo: {}, KL: {}'.format(epoch + 1, val_log['loss'][-1], val_log['neg_elbo'][-1], val_log['kl'][-1]))

            if self.summary_writer is not None:
                # Logging to file
                things_to_write = {}
                for key in val_log:
                    things_to_write[f'val/{key}_epoch'] = val_log[key][-1]
                for key in train_log:
                    things_to_write[f'train/{key}_epoch'] = np.mean(train_log[key][-100:])
                for key in things_to_write:
                    self.summary_writer.add_scalar(key, things_to_write[key], epoch + 1)

        return train_log, val_log

    def train_one_epoch(self):
        logs = ArrayDict()
        for iteration, data in enumerate(self.trainloader):
            self.global_step += 1
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

            if self.summary_writer and self.global_step % self.log_every_n_steps == 0:
                for key in log:
                    self.summary_writer.add_scalar(f'train/{key}_step', log[key], self.global_step)

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
            logs.append(log)
        # Mean over all data
        assert all([v.ndim == 1 for v in logs.values()])
        logs = {k: logs[k].mean() for k in logs}
        return logs
    


def attach_run_id(path, exp_name):
    # From stable-baselines-3
    max_run_id = 0
    path = pathlib.Path(path)
    for dir_path in path.glob(f'{exp_name}_[0-9]*'):
        prefix, _, suffix = dir_path.name.rpartition('_')
        if prefix == exp_name and suffix.isdigit() and int(suffix) > max_run_id:
            max_run_id = int(suffix)
    return f'{exp_name}_{max_run_id + 1}'



# from  https://github.com/rll/deepul/blob/master/deepul/utils.py
def savefig(fname, show_figure=True):
    os.makedirs(osp.dirname(fname), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
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
def save_results(samples, real_recon, vqvae_train_loss, vqvae_test_loss, prior_train_loss, prior_test_loss, result_dir, show_figure=False):
    samples, real_recon = samples.astype('float32'), real_recon.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_loss[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {prior_test_loss[-1]:.4f}')
    save_training_plot(vqvae_train_loss, vqvae_test_loss,'VQ-VAE Train Plot',
                       osp.join(result_dir, 'vqvae_train_plot.png'))
    save_training_plot(prior_train_loss, prior_test_loss,'PixelCNN Prior Train Plot',
                       osp.join(result_dir, 'prior_train_plot.png'))
    show_samples(samples, title='Samples',
                 fname=osp.join(result_dir, 'samples.png'))
    show_samples(real_recon, title='Reconstructions',
                 fname=osp.join(result_dir, 'reconstructions.png'))

