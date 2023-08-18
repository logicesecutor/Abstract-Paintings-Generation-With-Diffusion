# Set this path to be compatible with both linux and windows
ROOT_PATH = "F:\\Thesis\\Deep-Learning-Techniques-for-Image-Generation-from-Music"

import sys
sys.path.append(ROOT_PATH)

import torch, wget, os
from zipfile import ZipFile
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from modules.util import instantiate_from_config, makeDirectories, trainTestSubdivision, extractFile


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load(ROOT_PATH + "configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

if __name__ == "__main__":
    # Pytorch and pytorch-lightning settings
    device = "gpu" if torch.cuda.is_available() else "cpu"
    seed_everything(43) # Make the experiment reproducible
    torch.set_float32_matmul_precision('high')

    # Configuration file path and loading
    experiment_cfg_path = ROOT_PATH + "/configs/custom-ldm-cwa-vq-f8.yaml"
    config = OmegaConf.load(experiment_cfg_path)

    # Set all the path we need for the training
    dataset_name = config.data.dataset_name
    model_name = config.model.name

    n_classes = config.model.params.cond_stage_config.params["n_classes"]

    # Directory where to save generated samples
    sample_folder = ROOT_PATH + f"/sample/{model_name}/{dataset_name}"

    # Directory where to save check points
    checkpts_save_folder = ROOT_PATH + f"/model_checkpts/{model_name}/{dataset_name}"

    # Directory where to log
    logger_path = ROOT_PATH + "/logs" 

    # Directory of the Dataset
    dataset_path = ROOT_PATH + f"/Datasets/{dataset_name}"
    # Dataset extraction
    extractFile(ROOT_PATH + "/Datasets/absfig_filtered_tfr.zip", dataset_path)

    # Link to download the vq-Gan pretrained model with reduction factor of 8 (256x256 -> 32x32)
    vq_gan_pretrained_path = ROOT_PATH + "/pretrained_model/vq-f8"
    pretrained_url = "https://ommer-lab.com/files/latent-diffusion/vq-f8.zip"

    # Download Autoencoder pretrained model if does not exist
    if not os.path.isfile(vq_gan_pretrained_path + "/model.ckpt"):
      wget.download(pretrained_url, out = vq_gan_pretrained_path + "/vq-f8.zip")
      with ZipFile(vq_gan_pretrained_path + "/vq-f8.zip", "r") as zipObj:
        zipObj.extractall(vq_gan_pretrained_path)

    # Generate some utility Directories if they does not exixt
    makeDirectories((sample_folder, checkpts_save_folder, logger_path, vq_gan_pretrained_path))

    config.model.params.first_stage_config.params["ckpt_path"] = vq_gan_pretrained_path + "/model.ckpt"

    # Here we create the data and label files for train and test
    train_path, test_path, train_labels, test_labels = trainTestSubdivision(dataset_path)

    config.data.params.train.params["training_images_list_file"] = train_path
    config.data.params.train.params["train_labels_file"] = train_labels
    config.data.params.train.params["uncond_label"] = n_classes - 1

    config.data.params.validation.params["test_images_list_file"] = train_path
    config.data.params.validation.params["test_labels_file"] = test_labels
    config.data.params.validation.params["uncond_label"] = n_classes - 1

    # Batch size and Learning rate setted for the pytorch-lightning Trainer
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    n_gpus = config.lightning.trainer.n_gpus
    epochs = config.lightning.trainer.n_epochs

    # Model instantiation from the configuration file
    model = instantiate_from_config(config.model)

    if 'accumulate_grad_batches' in config.lightning.trainer:
        accumulate_grad_batches = config.lightning.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1

    config.lightning.trainer.accumulate_grad_batches = accumulate_grad_batches
    if config.model.scale_lr:
        model.learning_rate = accumulate_grad_batches * n_gpus * bs * base_lr
        print(f"Setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches} (accumulate_grad_batches) * {n_gpus} (num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)")
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    # Load the data module and the tensorboard logger for the training
    data = instantiate_from_config(config.data)
    logger = TensorBoardLogger(save_dir=logger_path, name="LDM_training")

    imageLogger = instantiate_from_config(config=config.lightning.callbacks.image_logger)

    trainer = pl.Trainer(default_root_dir= checkpts_save_folder,
                         max_epochs=epochs,
                         devices= 1, #n_gpus, 
                         accelerator="gpu" , 
                         deterministic=True,
                         callbacks= [imageLogger],
                         num_sanity_val_steps = 2,
                         logger=logger, 
                         #strategy="fsdp",
                         #precision="16-mixed", # Set To exploit the Tensor Cores if available
                         )
    trainer.fit(model, data)