import torch

from omegaconf import OmegaConf
import json
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from modules.util import instantiate_from_config, makeDirectories, trainTestSubdivision

ROOT_PATH = "/mnt/data1/bardella_data/gitRepos/Thesis/ldm_porting"
# Settings for the training

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    assert torch.cuda.is_available()
    seed_everything(43)
    
    experiment_cfg_path = ROOT_PATH + "/configs/custom_vqgan.yaml"
    config = OmegaConf.load(experiment_cfg_path)

    model_config = config["model"]
    data_config = config["data"]

    mode = "train" # "test"
    dataset_name = data_config.dataset_name
    model_name = model_config.name
    resize_image_size = data_config.params.train.params.size

    sample_folder = ROOT_PATH + f"/sample/{model_name}/{dataset_name}"
    checkpts_save_folder = ROOT_PATH + f"/model_checkpts/{model_name}/{dataset_name}"
    logger_path = ROOT_PATH 
    dataset_path = f"/mnt/data1/bardella_data/gitRepos/Thesis/Datasets/{dataset_name}"

    makeDirectories((sample_folder, checkpts_save_folder))
    train_path, test_path, _, _ = trainTestSubdivision(dataset_path)

    config.data.params.train.params["training_images_list_file"] = train_path
    config.data.params.validation.params["test_images_list_file"] = train_path

    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    n_gpus = config.trainer.n_gpus
    epochs = config.trainer.n_epochs

    vq_gan = instantiate_from_config(config.model)

    accumulate_grad_batches = 1
    vq_gan.learning_rate = accumulate_grad_batches * n_gpus * bs * base_lr
    
    data = instantiate_from_config(config.data)
    logger = TensorBoardLogger(save_dir=logger_path)

    trainer = pl.Trainer(default_root_dir= checkpts_save_folder,
                         #max_epochs=epochs,
                         devices=4, 
                         accelerator="cpu", 
                         deterministic=True, 
                         #enable_checkpointing=True, 
                         num_sanity_val_steps = 0,
                         #logger=logger
                         )

    trainer.fit(vq_gan, data)