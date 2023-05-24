import torch
from omegaconf import OmegaConf
import albumentations
from PIL import Image
from torchvision import transforms

import  numpy as np

from modules.util import instantiate_from_config
from tqdm import tqdm

device = "cuda:1" if torch.cuda.is_available() else None
assert device

def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(size, image_path):
        
        rescaler = albumentations.SmallestMaxSize(max_size = size)
        cropper = albumentations.CenterCrop(height=size,width=size)
        preprocessor = albumentations.Compose([rescaler, cropper])

        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image_transformed = preprocessor(image=image)["image"]
        image = (image_transformed/127.5 - 1.0).astype(np.float32)

        return (transforms.ToPILImage()(image_transformed), torch.tensor(image).permute(2, 0, 1))

def postprocess_image(img):
    img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    img = img.permute(1,2,0)
    img = img.to("cpu").numpy()
    img = (img * 255).astype(np.uint8)
    
    img = transforms.ToPILImage()(img)
    return img

ROOT_PATH = "/mnt/data1/bardella_data/gitRepos/Thesis/ldm_porting"

dataset_path = "/mnt/data1/bardella_data/gitRepos/Thesis/Datasets/wikiart"
test_data_file = dataset_path + "/test_data.txt"

# Pretrained model on Imagenet
check_point_path = ROOT_PATH + "/pretrained_model/vq-f8/model.ckpt"

# Pretrained model on Wikiart
check_point_path = ROOT_PATH + "/model_checkpts/vqgan/wikiart/vq-f8/last.ckpt"


experiment_cfg_path = ROOT_PATH + "/configs/custom_vqgan.yaml"
config = OmegaConf.load(experiment_cfg_path)


model = load_model_from_config(config=config.model, ckpt=check_point_path, device=device)


with open(test_data_file, "r") as test_fp:
    test_image_paths = [l for l in test_fp.read().splitlines()]

samples = None
average_loss = 0
with torch.no_grad():
    for i, test_image_path in tqdm(enumerate(test_image_paths[:samples])):

        original, image = preprocess_image(size=256, image_path=test_image_path)
        img_in = image.to(device=device).unsqueeze(dim=0)


        xrec, q_loss = model(img_in)
        aeloss, log_dict_ae = model.loss(q_loss, img_in, xrec, 0, model.global_step, last_layer=model.get_last_layer(), split="val")
        average_loss += aeloss
        xrec = xrec.squeeze(dim=0)

        img_out = postprocess_image(xrec)

        
        original.save(f"/mnt/data1/bardella_data/gitRepos/Thesis/ldm_porting/sample/vqgan/wikiart/original_{i}.png","png")
        img_out.save(f"/mnt/data1/bardella_data/gitRepos/Thesis/ldm_porting/sample/vqgan/wikiart/reconstructed_{i}.png","png")
    
    print(average_loss/(samples if samples is not None else len(test_image_paths[:samples]))  )