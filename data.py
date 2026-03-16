import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import PIL.Image as Image

image_size = 256
img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

class PolypGenFLDataset(Dataset):
    def __init__(self, data_path = '/home/khoi.ho/ML709/PolypGen2021_MultiCenterData_v3', center = 'all', transform=img_transform):
        assert center in ['all', '1', '2', '3', '4', '5' , '6'], "Center must be one of 'all', '1', '2', '3', '4', '5' or '6'"
        self.data_path = data_path
        self.center = center
        self.transform = transform
        self.image_size = 256
        self.image_paths, self.mask_paths = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = []
        mask_paths = []
        
        centers = ['1', '2', '3', '4', '5'] if self.center == 'all' else [self.center]
        for center in centers:
            center_path = os.path.join(self.data_path, f'data_C{center}')
            images_path = os.path.join(center_path, f'images_C{center}')
            masks_path = os.path.join(center_path, f'masks_C{center}')
            for img_name in os.listdir(images_path):
                img_path = os.path.join(images_path, img_name)
                mask_path = os.path.join(masks_path, img_name.replace('.jpg', '_mask.jpg'))

                image_paths.append(img_path)
                mask_paths.append(mask_path)
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.transform(image)

        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask