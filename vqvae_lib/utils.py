import os
import csv
import os.path as osp
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pathlib
import logging
from typing import Dict, Any, Optional

class CSVWriter:
    r"""
    From Pytorch Lighting

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_HPARAMS_FILE = "hparams.yaml"
    NAME_METRICS_FILE = "metrics.csv"

    def __init__(self, log_dir: str) -> None:
        self.hparams = {}
        self.metrics = []

        self.log_dir = log_dir
        # if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
        #     logging.warn(
        #         f"Experiment logs directory {self.log_dir} exists and is not empty."
        #         " Previous log files in this directory will be deleted when the new ones are saved!"
        #     )
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)


    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)


    def save(self) -> None:
        """Save recorded hparams and metrics into files."""

        if not self.metrics:
            return

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        with open(self.metrics_file_path, "w", newline="") as f:
            # Don't assign the writer to self.
            # Keeps an open reference and prevents pickling otherwise
            writer = csv.DictWriter(f, fieldnames=metrics_keys)
            writer.writeheader()
            writer.writerows(self.metrics)


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
        csv_writer=None,
        log_every_n_steps=5,
        on_epoch_end=None
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
      self.csv_writer = csv_writer
      self.on_epoch_end = on_epoch_end

    def train(self):
        val_log = ArrayDict()
        train_log = ArrayDict()
        val_log.append(self.validate())
        print('Initial loss:', val_log['loss'])
        for epoch in range(self.max_epochs):

            train_log.extend(self.train_one_epoch())
            val_log.append(self.validate())
            self.on_epoch_end(epoch)
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

                things_to_write = {}
                for key in val_log:
                    things_to_write[f'val/{key}_epoch'] = val_log[key][-1]
                for key in train_log:
                    things_to_write[f'train/{key}_epoch'] = np.mean(train_log[key][-100:])
            # Logging to file
            if self.summary_writer is not None:
                for key in things_to_write:
                    self.summary_writer.add_scalar(key, things_to_write[key], epoch + 1)
            if self.csv_writer is not None:
                self.csv_writer.log_metrics(things_to_write, step=epoch + 1)
                self.csv_writer.save()

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
def save_results(samples, real_recon, vqvae_train_loss, vqvae_test_loss, prior_train_loss=None, prior_test_loss=None, result_dir='./results', show_figure=False):
    samples, real_recon = samples.astype('float32'), real_recon.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_loss[-1]:.4f}')
    if prior_test_loss is not None:
        print(f'PixelCNN Prior Final Test Loss: {prior_test_loss[-1]:.4f}')
    save_training_plot(vqvae_train_loss, vqvae_test_loss,'VQ-VAE Train Plot',
                       osp.join(result_dir, 'vqvae_train_plot.png'))
    if not (prior_train_loss is None or prior_test_loss is None):
        save_training_plot(prior_train_loss, prior_test_loss,'PixelCNN Prior Train Plot',
                        osp.join(result_dir, 'prior_train_plot.png'))
    show_samples(samples, title='Samples',
                 fname=osp.join(result_dir, 'samples.png'))
    show_samples(real_recon, title='Reconstructions',
                 fname=osp.join(result_dir, 'reconstructions.png'))

