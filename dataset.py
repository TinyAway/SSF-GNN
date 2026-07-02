from torch.utils.data import Dataset
import torch
import numpy as np
from osgeo import gdal
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
from Compute_indices import compute_indices


class_dict = ["Normal", "Wildfire", "Flood", "Oilspill", "Redtide", "Volcaniceruption", "Algalbloom"]


class DatasetAnomaly(Dataset):
    def __init__(self, datapath, transform=False, load_in_memory=False, use_indices=False, binary_classify=False):
        self.transform = transform
        self.use_indices = use_indices
        self.image_list = []
        self.class_list = []
        self.load_in_memory = load_in_memory
        for i, cls_name in enumerate(class_dict):
            for file_name in os.listdir(os.path.join(datapath, cls_name)):
                if file_name[-4:] == '.tif':
                    if load_in_memory:
                        img = gdal.Open(os.path.join(datapath, cls_name + '/' + file_name))
                        img = img.ReadAsArray().astype(np.float32)  # 获取数据
                        img = (img - 1000) / 10000
                        img = np.clip(img, -0.1, 1.0)
                        self.image_list.append(img)
                    else:
                        self.image_list.append(os.path.join(datapath, cls_name + '/' + file_name))
                    if binary_classify:
                        self.class_list.append(int(i>0))
                    else:
                        self.class_list.append(i)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        label = self.class_list[item]
        if self.load_in_memory:
            img = self.image_list[item]
        else:
            img = gdal.Open(self.image_list[item])
            img = img.ReadAsArray().astype(np.float32) # 获取数据
            img = (img - 1000) / 10000
            img = np.clip(img, -0.1, 1.0)
        if self.transform:
            img = np.ascontiguousarray(random_transform(img))
        if self.use_indices:
            indices = compute_indices(img)
            img = self.Normalize(img)
            img = np.concatenate([img, indices], axis=0)
        else:
            img = self.Normalize(img)
        return img, label

    def Normalize(self, img):
        mean = np.array([0.0736, 0.0802, 0.0935, 0.0933, 0.1120, 0.1290, 0.1386, 0.1369, 0.1398,
                        0.1532, 0.1195, 0.0972], dtype=np.float32)
        std = np.array([0.1041, 0.1055, 0.1128, 0.1293, 0.1377, 0.1508, 0.1594, 0.1627, 0.1651,
                        0.1768, 0.1612, 0.1433], dtype=np.float32)
        img = (img - mean[:, None, None]) / std[:, None, None]
        return img


# compute mean and variance values of the dataset
def get_mean_std(loader):
    # Var[x] = E[X**2]-E[X]**2
    mean_sum = torch.zeros(12)
    std_sum = torch.zeros(12)
    num_batches = 0
    for data, name in tqdm(loader):
        mean_sum += torch.mean(data, dim=[0, 2, 3])
        std_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    print(num_batches)
    print(mean_sum)
    mean = mean_sum / num_batches
    std = (std_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def random_transform(image):
    flip = random.random()
    rotate = random.random()
    if flip > 0.5:
        image = random_flip(image)
    if rotate > 0.5:
        image = random_rotate(image)
    return image

def random_flip(image):
    mode = random.choice([1, 2])
    new_image = np.flip(image, mode)
    return new_image

def random_rotate(image):
    mode = random.choice([1, 2, 3])
    new_image = np.rot90(image, mode, (1, 2))
    return new_image

def to_rgb(img):
    mean = np.array([0.0736, 0.0802, 0.0935, 0.0933, 0.1120, 0.1290, 0.1386, 0.1369, 0.1398,
                     0.1532, 0.1195, 0.0972], dtype=np.float32)
    std = np.array([0.1041, 0.1055, 0.1128, 0.1293, 0.1377, 0.1508, 0.1594, 0.1627, 0.1651,
                    0.1768, 0.1612, 0.1433], dtype=np.float32)
    img = img * std[:, None, None] + mean[:, None, None]
    return img



if __name__ == "__main__":
    path = "E:/datasets/SEN2MHD/train"
    dataset = DatasetAnomaly(path, transform=True, load_in_memory=False, use_indices=False, binary_classify=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for img, lbl in dataloader:
        print(img.shape, lbl.shape)












