import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import natsort

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

class Dataset:
    '''
    Dateset: custom dataset
    transform and return datasets -> test, train
    '''

    def __init__(self):
        super(Dataset, self).__init__()
        self._transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def train_loader(self):
        '''
        :return: train_loader
        '''

        #train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self._transform)
        # train_set = torchvision.datasets.ImageFolder(root='./data', transform=self._transform, target_transform='None')
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
        train_set = CustomDataSet('./data/train', transform=self._transform)
        train_loader = DataLoader(train_set, batch_size=10, shuffle=False,
                                       num_workers=4, drop_last=True)
        return train_loader

    def test_loader(self):
        '''
        :return: test_loader
        '''

        test_set = CustomDataSet('./data/valid', transform=self._transform)
        test_loader = DataLoader(test_set, batch_size=10, shuffle=False,
                                  num_workers=4, drop_last=True)
        return test_loader


Dataset = Dataset()
