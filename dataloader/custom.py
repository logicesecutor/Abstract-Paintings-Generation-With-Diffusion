from torch.utils.data import Dataset
from dataloader.base import ImagePaths
from PIL import Image
import json
import albumentations
import numpy as np


# class CustomBase(Dataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.data = None

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         example = self.data[i]
#         return example


# class CustomTrain(CustomBase):
#     def __init__(self, size, training_images_list_file):
#         super().__init__()
#         with open(training_images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         self.data = ImagePaths(paths=paths, size=size, random_crop=False)


# class CustomTest(CustomBase):
#     def __init__(self, size, test_images_list_file):
#         super().__init__()
#         with open(test_images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         self.data = ImagePaths(paths=paths, size=size, random_crop=False)

# class CustomTrain(CustomBase):
#     def __init__(self, size, training_images_list_file, train_labels_file):
#         super().__init__()
#         with open(training_images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         with open(train_labels_file, "r") as f:
#             labels_paths = f.read().splitlines()
#             labels_paths = [int(l) for l in labels_paths]

#         self.data = ImagePaths(paths=paths, size=size, labels=labels_paths, random_crop=False)


# class CustomTest(CustomBase):
#     def __init__(self, size, test_images_list_file, test_labels_file):
#         super().__init__()
#         with open(test_images_list_file, "r") as f:
#             paths = f.read().splitlines()
#         self.data = ImagePaths(paths=paths, size=size, labels=test_labels_file, random_crop=False)

class WikiartImageDataset(Dataset):
    def __init__(self, size, uncond_label, random_crop=False):
        super().__init__()

        self.image_paths =  None
        self.labels = None
        self.size = size
        self.random_crop = random_crop
        self.uncond_label = uncond_label

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return len(self.labels)
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, idx):
        image = self.preprocess_image(self.image_paths[idx])
        label = self.labels[idx]
        
        return {"image":image, 
                "class_label":label, 
                "unconditional_label":self.uncond_label
                }
    
    


class WikiartImageDatasetTrain(WikiartImageDataset):
    def __init__(self, size, uncond_label, training_images_list_file, train_labels_file, random_crop=False, transform=None):
        super().__init__(size, uncond_label, random_crop=random_crop)
        with open(training_images_list_file, "r") as f:
            self.image_paths = f.read().splitlines()
        with open(train_labels_file, "r") as f:
            self.labels = [int(l) for l in f.read().splitlines()]


class WikiartImageDatasetTest(WikiartImageDataset):
    def __init__(self, size, uncond_label, test_images_list_file, test_labels_file, random_crop=False, transform=None):
        super().__init__(size, uncond_label, random_crop=random_crop)

        with open(test_images_list_file, "r") as f:
            self.image_paths = f.read().splitlines()
        with open(test_labels_file, "r") as f:
            self.labels = [int(l) for l in f.read().splitlines()]
