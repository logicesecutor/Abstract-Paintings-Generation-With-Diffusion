# Deep-Learning-Techniques-for-Image-Generation-from-Music
This project was born with the aim of understanding and implementing a Diffusion model capable of generating abstract paintings starting from a classical music audio source.

<img src="https://github.com/logicesecutor/Deep-Learning-Techniques-for-Image-Generation-from-Music/blob/main/src/images/orange.png" alt="Generated Abstract Painting" width="500"/>

Diffusion Pipeline implementation for Abstract-Art image generation using a class-conditioned Latent Diffusion Model. 
All the used models are highly based on the models in the following repos:
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) for the U-Net denoiser.
- [Taming Tranformers](https://github.com/CompVis/taming-transformers) for the VQ-GAN and pre-trained models.


## Clone a private repo
This is a private repo so to clone you need to have the PAT. 
Once you have it the correct bash command is:
```bash
git clone https://[PAT]@github.com/[username]/reponame.git some/dest/dir
```

## Environment settings
### Setting a local environment
To perfectly reproduce the environment on a local system, you should use a conda environment to run the model. 
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
In order to make it work on colab we should install those package version

```bash
!pip install pytorch-lightning==1.9.4
!pip install einops
!pip install omegaconf
!pip install kornia
!pip install wget
```

## The Dataset preparation
The used dataset is a custom version of Wikiart subdivided into 8 different color labels.
The dataset needs to be downloaded and the zip file must be put in the Datasets directory.
When the **"LDM_train.py"** file will extract all the dataset and generate the necessary training files.

[Dataset Download link](https://drive.google.com/file/d/1LSfJZ6IAWbCi8jAQJ2IHV9afUbwFbZ4V/view?usp=drive_link)

## The Configuration file
In the configs folder, there are some configuration files that contain the settings for the training phase.
From here we can set the VQGAN, the diffusion model, and manage all the Dataset directories.

In order to give the net our custom dataset we must provide 2 files txt that contain the path list to single images and 2 file txt which include the categorical labels corresponding to the images in this format:

```
$path/to/dataset/train.txt
path_to_image_1.txt
path_to_image_2.txt
.....
path_to_image_n.txt
```
```
$path/to/dataset/train_labels.txt
label_image_1.txt
label_image_2.txt
.....
label_image_n.txt
```
We should have a 'train.txt', 'train_labels.txt', and a 'validation.txt', 'validation_labels.txt' files.

In the dataloader folder, we have a custom file where we can modify the behavior of how we load the data and what pre-process we want to apply.

# Two stages
We have two stages:
- The first one is the Encoder which can be trained in the **VQGAN_train.py** script.
- The second one is the Diffusion phase which can be trained using the **LDM_train.py** script.
  
## Training and model settings
Every setting of the model, the number of GPU, the epoch, and the Batch size can be modified in the config_file.yaml in the config folder:
- ***"custom_vqgan.yaml"*** for the first stage.
- ***"custom-ldm-cwa-vq-f8.yaml"*** for the second stage

## Pre-trained model
If we want to use a pre-trained model and pass it to the trainer we should provide the model directory path to the trainer fit function.

```python
ckpt_path = "/path/to/pre-trained/model.ckpt"
trainer = pl.Trainer(...)
trainer.fit(model,
            data,
            ckpt_path=ckpt_path
)
```

For the first stage I have used the vq-gan f=8, VQ (Z=16384, d=4) from latent diffusion:

[VQ-GAN f=8 Download link](https://ommer-lab.com/files/latent-diffusion/vq-f8.zip)

Other pre-trained models can be downloaded from the official implementation of latent diffusion and taming transformers.

The second stage was trained from scratch and here I leave the download to my trained model:

This model was trained for a total of 300 epochs on an A100 40GB with a batch size of 64.

[LDM-300 Download link](https://drive.google.com/file/d/13NikX84LivRciepkZB2mi5vZI-183ZZ8/view?usp=drive_link)

## NOTE!
At the start of each script, we should change the ***ROOT_PATH*** Variable to make the script work both on Linux and Windows and be easily transferable to a notebook-style script.

For each train script, there is also the relative inference script.
