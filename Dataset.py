

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageOps
from Params import Params
import torchvision.transforms.functional as TF
import torchvision.transforms as trans
import os
import time
import random
import natsort

class PairDataset(Dataset):
    def __init__(self, input_path, target_path, train=True):
        self.input_path = input_path
        self.target_path = target_path
        all_imgs_in = os.listdir(input_path)
        self.total_imgs_in = natsort.natsorted(all_imgs_in)
        # print("self.total_imgs_in " + str(self.total_imgs_in))
        all_imgs_out = os.listdir(target_path)
        self.total_imgs_out = natsort.natsorted(all_imgs_out)
        # print("self.total_imgs_out " + str(self.total_imgs_out))

    def transform(self, image_input, image_output):
        # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image_input = resize(image_input)
        # image_output = resize(image_output)

        width, height = image_input.size

        # image_input.save("tr_in.png")
        # print("image_input type: " + str(type(image_input)))
        # Color jitter
        if random.random() > 0.25:
            transform = trans.Compose([
            trans.ColorJitter(brightness=0.15, contrast = 0.15, saturation = 0.15, hue = 0.05),
            ])
            image_input = transform(image_input)

        # Gaussian blurr
        if random.random() > 0.5:
            kernel_size = random.randrange(1,9,2)#start, stop, step - odd numbers only
            sigma =  random.random()*5
            image_input = TF.gaussian_blur(image_input, kernel_size=kernel_size, sigma=sigma)

        # Random horizontal flipping
        if random.random() > 0.5:
            image_input = TF.hflip(image_input)
            image_output = TF.hflip(image_output)

        # Random vertical flipping
        if random.random() > 0.5:
            image_input = TF.vflip(image_input)
            image_output = TF.vflip(image_output)

        # random noise
        # if random.random() > 0.5:
            # sigma=random.uniform(0.01, 0.1)
            # image_input = trans.random_noise(image_input,var=sigma**2)

        # image_input.save("tr_out.png")

        # Transform to tensor
        image_input = TF.to_tensor(image_input)
        image_output = TF.to_tensor(image_output)
        return image_input, image_output

    def __getitem__(self, index):
        img_loc = os.path.join(self.input_path, self.total_imgs_in[index])
        # print(img_loc)
        image_input = Image.open(img_loc)
        img_loc = os.path.join(self.target_path, self.total_imgs_out[index])
        # print(img_loc)
        image_output = Image.open(img_loc)
        x, y = self.transform(image_input, image_output)
        return x, y

    def __len__(self):
        return len(self.total_imgs_in)

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
