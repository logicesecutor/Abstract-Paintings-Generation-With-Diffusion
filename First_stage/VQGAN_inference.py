#@title loading utils
import torch
from omegaconf import OmegaConf
from torchvision import utils

from modules.util import instantiate_from_config
from setup import getModelSettings
from tqdm import tqdm
from dataloader.dataloader import DataLoaderManager

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    config = OmegaConf.load("/home/ilchrees/data/gitRepo/Thesis/vqgan_pipeline/configs/config-vq-f8.yaml")  
    model = load_model_from_config(config, "/home/ilchrees/data/gitRepo/Thesis/vqgan_pipeline/model_checkpts/vqgan/wikiart/model.ckpt")
    return model


experiment_cfg_path = "/home/ilchrees/data/gitRepo/Thesis/vqgan_pipeline/configs/config-vq-f8.yaml"
config = OmegaConf.load(experiment_cfg_path)

model_config = config["model"]
data_config = config["data"]

dataset_name = data_config.dataset_name
model_name = model_config.name
resize_image_size = data_config.params.train.params.size

sample_folder = f"/home/ilchrees/data/gitRepo/Thesis/vqgan_pipeline/sample/{model_name}/{dataset_name}"
checkpts_save_folder = f"/home/ilchrees/data/gitRepo/Thesis/vqgan_pipeline/model_checkpts/{model_name}/{dataset_name}"

dataset_path = f"/home/ilchrees/data/gitRepo/Thesis/Datasets/{dataset_name}"


vqgan = get_model()
vqgan.eval()

model_settings, dataset_settings = getModelSettings(model_name, dataset_name)
dlm = DataLoaderManager(
            dataset_name=dataset_name, 
            path= dataset_path, 
            model_name=model_name,
            image_size=model_settings["image_size"], 
            batch_size=model_settings["batch_size"],
            shuffle=True,
        )

dataloader = dlm.getDataLoader(split_train_test=True, test=True) 

sample_size = 2

dataloader = tqdm(dataloader)

for i, (img, label) in enumerate(dataloader):
    vqgan.zero_grad()
    img = img.to(device)
    
    sample = img[:sample_size]

    with torch.no_grad():
        out, _ = vqgan(sample)

    final= torch.cat([sample, out], 0)
    utils.save_image(
        final,
        sample_folder + f"/{str(i).zfill(5)}.png",
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )
    
    break
print()