# ift-6269-vq-vae

## Dependencies

```sh
conda create -n vqvae python=3.9
conda activate vqvae
pip install -r requirements.txt
```

You will also want to install tensorboard

## Run

```sh
RESULT_DIR='./result_dir'
DATA_DIR='./data_dir'
python scripts/run_vqvae_dl_cifar10.py --result_dir $RESULT_DIR --data_dir $DATA_DIR
```

Everything (model, metrics, tensorboard event, images...) will be saved under `$RESULT_DIR/vqvae_dl_cifar10_{run_id}`, where `{run_id}` will be `1` for your first run.

View tensorboard:

```
tensorboard --logdir $RESULT_DIR --port 8848 --bind_all
```

After the run is finished, you can find reconstruction and samples in `$RESULT_DIR/vqvae_dl_cifar10_{run_id}`

Similar for `run_vqvae_mse_cifar10.py`.



Ignore follwoing parts
> ## Using Colab

> In Colab, try to open the notebook `demo.ipynb` from Github (you need to check the option to include private repos). Then, zip the project, name it `ift-6269-vq-vae.zip` and upload it. Then you can run it.


> ## Read the code



> In `vqvae.py`, you should first try to understand the code for these classes:

> * `VQVAEBase`: VQ-VAE encoder and decoder, without prior
> * `VQVAEPrior`: VQ-VAE prior, a PixelCNN. You can also read the PixelCNN code, but it is a bit complicated, because we need masked convolution. Just realize that it is a PixelCNN is enough. * 
> * `VQVAE`: after both `VQVAEBase` and `VQVAEPrior` are trained, we combined them in `VQVAE`. The only purpose of this class is to generate samples and do reconstruction.

> Other classes in this file are just building blocks (e.g. residual blocks, masked or not).

> `main.py` implements the training. You can only look at `main()`. Other code are just logging and visualization. What it does:

> * Train the encoder and decoder (`VQVAEBase`).
> * After that, use the encoder to generate a dataset of indices of the embeddings.
> * Train the prior (`VQVAEPrior`) on this indices dataset.
> * Combine `VQVAEBase` and `VQVAEPrior` into `VQVAE`. We can then generate samples and reconstruct inputs.


> ## Notes

> * Currently using CIFAR10. Note there is no validation split. We should create it.
> * Most plotting and visualization code is from this [repo](https://github.com/rll/deepul). They are pretty primitive and we just implement tensorboard logging later.
