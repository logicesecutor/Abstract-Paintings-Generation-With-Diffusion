import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.vqgan import VQModel
from modules.util import instantiate_from_config, makeDirectories, trainTestSubdivision

ROOT_PATH = "/mnt/data1/bardella_data/gitRepos/Thesis/ldm_porting"
# Settings for the training

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    assert torch.cuda.is_available()
    seed_everything(43)
    
    experiment_cfg_path = ROOT_PATH + "/configs/custom_vqgan.yaml"
    config = OmegaConf.load(experiment_cfg_path)


    dataset_name = config.data.dataset_name
    model_name = config.model.name

    sample_folder = ROOT_PATH + f"/sample/{model_name}/{dataset_name}"
    checkpts_save_folder = ROOT_PATH + f"/model_checkpts/{model_name}/{dataset_name}"
    logger_save_folder = ROOT_PATH + "/logs"
     
    dataset_path = f"/mnt/data1/bardella_data/gitRepos/Thesis/Datasets/{dataset_name}"

    makeDirectories((sample_folder, checkpts_save_folder))
    train_path, test_path, train_labels, test_labels = trainTestSubdivision(dataset_path)

    config.data.params.train.params["training_images_list_file"] = train_path
    config.data.params.train.params["train_labels_file"] = train_labels

    config.data.params.validation.params["test_images_list_file"] = train_path
    config.data.params.validation.params["test_labels_file"] = test_labels

    # Batch size and Learning rate setted for the pytorch-lightning Trainer
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    n_gpus = config.lightning.trainer.n_gpus
    epochs = config.lightning.trainer.n_epochs

    # Model instantiation from the configuration file
    model = VQModel(**config.model.params)

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

    imageLogger = instantiate_from_config(config.lightning.callbacks.image_logger)
    tb_logger = TensorBoardLogger(save_dir=logger_save_folder,
                                  name="VQGAN_training",
                                  )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        save_last=True,
        monitor=config.model.params.monitor,
        mode="min",
        dirpath="/mnt/data1/bardella_data/gitRepos/Thesis/ldm_porting/model_checkpts/vqgan/wikiart/vq-f8",
        filename="VQGAN-hierarchical-{epoch:02d}-{aeloss}",
    )


    trainer = pl.Trainer(default_root_dir= checkpts_save_folder,
                         max_epochs=epochs,
                         devices=n_gpus, 
                         accelerator="gpu", 
                         #strategy="ddp_find_unused_parameters_true",
                         deterministic=True, 
                         callbacks= [checkpoint_callback, imageLogger],
                         num_sanity_val_steps = 2,
                         logger=[tb_logger],
                         )

    trainer.fit(model,
                data,
                ckpt_path="/mnt/data1/bardella_data/gitRepos/Thesis/ldm_porting/model_checkpts/vqgan/wikiart/vq-f8/VQGAN-hierarchical-epoch=32-rec_loss=0.ckpt"
    )