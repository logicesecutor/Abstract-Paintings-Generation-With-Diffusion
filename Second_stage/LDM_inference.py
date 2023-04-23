import torch
from omegaconf import OmegaConf
import os
from modules.util import instantiate_from_config
from models.diffusion.ddim import DDIMSampler
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

ROOT_PATH = "/mnt/data1/bardella_data/gitRepos/Thesis/ldm_porting"


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    device = torch.device("cuda:2")
    model.to(device)
    model.eval()

    return model


config = OmegaConf.load(ROOT_PATH + "/configs/custom-ldm-cwa-vq-f8.yaml")  
sample_folder = ROOT_PATH + f"/sample/ldm/wikiart"

vq_gan_pretrained_ckpt_path = ROOT_PATH + "/pretrained_model/vq-f8/model.ckpt"
ldm_pretrained_ckpt_path = ROOT_PATH + "/model_checkpts/ldm/wikiart/epoch=136-step=31784.ckpt"
config.model.params.first_stage_config.params["ckpt_path"] = vq_gan_pretrained_ckpt_path

model = load_model_from_config(config, ldm_pretrained_ckpt_path)
sampler = DDIMSampler(model)

# Quality, sampling speed and diversity are best controlled via the scale,
# ddim_steps and ddim_eta variables. As a rule of thumb, higher values of 
# SCALE produce better samples at the cost of a reduced output diversity. 
# Furthermore, increasing ddim_steps generally also gives higher quality samples, 
# but returns are diminishing for values > 250. Fast sampling (i e. low values of ddim_steps)
# while retaining good quality can be achieved by using ddim_eta = 0.0.

classes = [0,5,6,8]   # define classes to be sampled here
n_samples_per_class = 3
unconditional_class = 8

ddim_steps = 100 
ddim_eta = 0.5
scale = 3  # for unconditional guidance

all_samples = list()

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(n_samples_per_class*[unconditional_class]).to(model.device)})
        
        
        for class_label in classes:
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class*[class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples_per_class,
                                             shape=[4, 32, 32],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc, 
                                             eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                         min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)


# display as grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=n_samples_per_class)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
img = Image.fromarray(grid.astype(np.uint8))


# Save the producted image
image_name = f"sample_steps-{ddim_steps}_scale-{scale}_labels_-{','.join(str(n) for n in classes)}_eta-{ddim_eta}_uc-{unconditional_class}"
image_ext = ".png"
image_number = [0]
for entry in os.listdir(sample_folder):
    if os.path.isfile(os.path.join(sample_folder, entry)):
        splitted_entry = entry[:-len(image_ext)].split("_")
        number = int(splitted_entry.pop(-1))
        if splitted_entry == image_name.split("_"):
            image_number.append(number)

img.save(sample_folder+"/"+image_name+"_"+str(max(image_number) + 1)+image_ext)