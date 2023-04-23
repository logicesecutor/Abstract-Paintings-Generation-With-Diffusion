# Deep-Learning-Techniques-for-Image-Generation-from-Music
Diffusion Pipeline implementation for Abstract-Art image generation using a class-conditioned Latent Diffusion Models. 
All the used model are highly based on the models in the following repos:
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) for the U-Net denoiser.
- [Taming Tranformers](https://github.com/CompVis/taming-transformers) for the VQ-GAN model and pretrained model.


## Clone a private repo
This is a private repo so to clone you need to have the PAT. 
Once you have it the correct bash command is:
```bash
git clone https://[PAT]@github.com/[username]/reponame.git some/dest/dir
```

## Environment settings
### Setting a local environment
To perfectly replicate the environment on a local system I suggest to use a conda environment to run the model. 
I provide the repo with a environment.yaml from which we can import the dependencies:

```bash
conda env create -f environment.yaml
conda activate thesis
```

Below here, I also reported all the bash/shell command that I used to create the environment during the development. They should be ran in sequence:

```bash
conda create -n nameOfEnv python=3.9
conda activate nameOfEnv

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c - pytorch -c nvidia
conda install -c conda-forge pytorch-lightning==2.0.0
conda install -c conda-forge omegaconf==2.3.0
conda install -c anaconda scikit-learn
conda install -c conda-forge einops==0.6.0
conda install -c conda-forge tensorboard
pip install albumentations=1.3.0
pip install wget
```
### Setting a CoLab environment

## The Dataset preparation

## The Configuration file
In the configs folder there are some configuration files:


