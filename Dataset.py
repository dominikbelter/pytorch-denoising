

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageOps
from Params import Params
import torchvision.transforms.functional as TF
import os
import random
import natsort

class PairDataset(Dataset):
    def __init__(self, input_paths, target_paths, train=True):
        self.input_paths = input_paths
        self.target_paths = target_paths

    def transform(self, image_input, image_output):
        # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image_input = resize(image_input)
        # image_output = resize(image_output)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image_input, output_size=(512, 512))
        image_input = TF.crop(image_input, i, j, h, w)
        image_output = TF.crop(image_output, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image_input = TF.hflip(image_input)
            image_output = TF.hflip(image_output)

        # Random vertical flipping
        if random.random() > 0.5:
            image_input = TF.vflip(image_input)
            image_output = TF.vflip(image_output)

        # Transform to tensor
        image_input = TF.to_tensor(image_input)
        image_output = TF.to_tensor(image_output)
        return image_input, image_output

    def __getitem__(self, index):
        image = Image.open(self.input_paths[index])
        image_output = Image.open(self.target_paths[index])
        x, y = self.transform(image, image_output)
        return x, y

    def __len__(self):
        return len(self.image_paths)

class CustomDataSet(Dataset):
    def __init__(self, main_dir, channels_in, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.channels_in = channels_in
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.channels_in == 1 :
            gray_image = ImageOps.grayscale(image)
            tensor_image = self.transform(gray_image)
        else :
            tensor_image = self.transform(image)
        return tensor_image

class Dataset(Params):
    '''
    Dateset: custom dataset
    transform and return datasets -> test, train
    '''

    def __init__(self):
        super(Dataset, self).__init__()
        self._transform = transforms.Compose(
            [transforms.ToTensor()])
            #[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def train_loader(self, path): #project: added argument path
        '''
        :return: train_loader
        '''

        train_set = CustomDataSet(path, channels_in=self.channels_in, transform=self._transform)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                       num_workers=1, drop_last=True)
        self.num_train_images = train_set.__len__()
        print('Number of loaded train images: ' + str(self.num_train_images))
        return train_loader

    def train_loader_img2img(self, path_input, path_output): #project: added argument path
        '''
        :return: train_loader
        '''

        train_set_input = CustomDataSet(path_input, channels_in=self.channels_in, transform=self._transform)
        train_set_output = CustomDataSet(path_output, channels_in=self.channels_in, transform=self._transform)
        train_loader_input = DataLoader(train_set_input, batch_size=self.batch_size, shuffle=False,
                                       num_workers=1, drop_last=True)
        train_loader_output = DataLoader(train_set_output, batch_size=self.batch_size, shuffle=False,
                                       num_workers=1, drop_last=True)
        self.num_train_images = train_set_input.__len__()
        if train_set_input.__len__() != train_set_output.__len__() :
            print("Number of input and output images does not match")
        print('Number of loaded training images: ' + str(self.num_train_images))
        return zip(train_loader_input, train_loader_output)

    def train_loader_pair(self, path_input, path_output): #project: added argument path
        '''
        :return: train_loader
        '''

        train_set = PairDataset(path_input, path_output)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                       num_workers=1, drop_last=True)
        self.num_train_images = train_set.__len__()
        print('Number of loaded training images: ' + str(self.num_train_images))
        return train_loader

    def test_loader(self, path): #project: added argument path
        '''
        :return: test_loader
        '''
        
        test_set = CustomDataSet(path, channels_in=self.channels_in, transform=self._transform)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,#project: read all files from the folder
                                  num_workers=0, drop_last=False)
        self.num_test_images = test_set.__len__()
        print('Number of loaded test images: ' + str(self.num_test_images))
        return test_loader


Dataset = Dataset()
