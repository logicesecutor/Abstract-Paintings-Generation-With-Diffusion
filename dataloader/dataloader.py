from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Lambda, Resize, ToPILImage, CenterCrop, Normalize
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import json, sys
import torch


class WikiartImageDataset(Dataset):
    def __init__(self, path:str, transform=None):
        with open("/".join([path,"dataset.json"])) as flabels:
            img_labels = json.load(flabels)
        
        self.labels = [label[1] for label in img_labels["labels"]]
        self.img_dir = [path+"/"+label[0] for label in img_labels["labels"]]
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        image = Image.open(self.img_dir[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Dataloder for the cleaned wikiart dataset
# class WikiartImageDataset():
#     def __init__(self, path, transform=None):
#         self.dataset_path = path

#         with open(path+"dataset.json", "r") as dataset:
#             self.dataset_info = json.load(dataset)["labels"]

#         with open(path+"mapping.json", "r") as mapping:
#             self.label_mapping = json.load(mapping)

#         # Transform the mapping from list into the dictionary format as
#         # {k:textual_label, v:numerical_label}
#         self.label_mapping = {k:v for k, v in self.label_mapping}
        
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset_info)
    

#     def __getitem__(self, idx):
#         img_folder = self.dataset_info[idx][0]
#         image = Image.open(self.dataset_path+img_folder)
#         label_txt = img_folder.split('/')[0]
#         label = self.label_mapping[label_txt]

#         if self.transform:
#             image = self.transform(image)
        
#         return image, label


class AlphabetFontImageDataset(Dataset):
    def __init__(self, path, transform=None):
        
        self.transform = transform
        self.img_dir, self.labels = self.load_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(np.uint8(self.img_dir[idx])).convert('RGB')
        label = int(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def load_data(self, path):
        
        temp = np.load(path)
        return temp['images'], temp['labels']



class DataLoaderManager():
    def __init__(self, dataset_name, path, model_name, image_size, batch_size, shuffle=True):

        self.transform = None
        self.model_name=model_name
        self.path=path

        self.image_size=image_size
        self.batch_size=batch_size
        self.shuffle=shuffle

        # Define image transformations (e.g. using torchvision)
        self.transform = Compose([
                    Resize(size=(image_size, image_size)),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Lambda(lambda t: (t * 2) - 1)
        ])

        self.reverse_transform = Compose([
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            Lambda(lambda t: t * 255.),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ])

        if dataset_name=="alphabet":
            self.dataset = AlphabetFontImageDataset(path=path, transform=self.transform)
        elif dataset_name=="wikiart":
            self.dataset = WikiartImageDataset(path=path, transform=self.transform)
        else:
            print("Dataset name not available", sys.stderr)
            return
        
        


    def getDataLoader(self, split_train_test=False, division=[0.8, 0.2], train=False, test=False, generator_seed=42):

        if split_train_test:
            self.dataset_train, self.dataset_test = random_split(self.dataset, division, generator=torch.Generator().manual_seed(generator_seed))

            if train:
                dataset = self.dataset_train
            elif test:
                dataset = self.dataset_test
            else:
                print("Train or Set subset must be selected", file=sys.stderr)
                return
        else:
            dataset=self.dataset

        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)