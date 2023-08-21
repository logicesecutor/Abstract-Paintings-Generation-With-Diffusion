# Deep-Learning-Techniques-for-Image-Generation-from-Music
Diffusion Pipeline implementation for Abstract-Art image generation using a class-conditioned Latent Diffusion Model. 
All the used models are highly based on the models in the following repos:
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) for the U-Net denoiser.
- [Taming Tranformers](https://github.com/CompVis/taming-transformers) for the VQ-GAN model and pre-trained model.


## Clone a private repo
This is a private repo so to clone you need to have the PAT. 
Once you have it the correct bash command is:
```bash
git clone https://[PAT]@github.com/[username]/reponame.git some/dest/dir
```

## Environment settings
### Setting a local environment
To perfectly reproduce the environment on a local system, I think you should use a conda environment to run the model. 
I provide the repo with an environment.yaml from which we can import the dependencies:

```bash
conda env create -f environment.yaml
conda activate thesis
```

Below, I also reported all the bash/shell commands I used to create the environment during the development. They should be run in sequence:

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
The used dataset is a custom version of Wikiart subdivided into 8 different color labels.
The dataset needs to be downloaded and the zip file must be put in the Datasets directory.
When the **"LDM_train.py"** file will extract all the dataset and will generate the necessary files for the training.

[Dataset Download link](https://github.com/CompVis/latent-diffusion)

## The Configuration file
In the configs folder, there are some configuration files that contain the settings for the training phase.
From here we can set the VQGAN, the diffusion model, and manage all the Dataset directories.

In order to give the net our custom dataset we must provide 2 files txt that contain the path list to single images and 2 file txt which include the categorical labels corresponding to the images in this format:

```
$path/to/dataset/train.txt
path_to_image_1.txt
path_to_image_2.txt
path_to_image_3.txt
path_to_image_4.txt
.....
path_to_image_n.txt
```
```
$path/to/dataset/train_labels.txt
label_image_1.txt
label_image_2.txt
label_image_3.txt
label_image_4.txt
.....
label_image_n.txt
```
We should have a 'train.txt', 'train_labels.txt', and a 'validation.txt', 'validation_labels.txt' files.

In the data loader folder, we have a custom file where we can modify the behavior of how we load the data and what pre-process we want to apply them.
