import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageOps
import os
import natsort

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
        # image.save(results_folder + f"/{idx + 1}loaded_pil_color_image.png")
        # gray_image.save(results_folder + f"/{idx + 1}loaded_pil_image.png")
        # save_image(tensor_image, results_folder + f"/{idx + 1}loaded_image.png")
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

        #train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self._transform)
        # train_set = torchvision.datasets.ImageFolder(root='./data', transform=self._transform, target_transform='None')
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
        train_set = CustomDataSet(path, channels_in=self.channels_in, transform=self._transform)
        train_loader = DataLoader(train_set, batch_size=10, shuffle=False,
                                       num_workers=1, drop_last=True)
        return train_loader

    def test_loader(self, path): #project: added argument path
        '''
        :return: test_loader
        '''
        
        test_set = CustomDataSet(path, channels_in=self.channels_in, transform=self._transform)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,#project: read all files from the folder
                                  num_workers=0, drop_last=False)
        return test_loader


Dataset = Dataset()

# for i, data_in in enumerate(Dataset.train_loader(train_folder_input)):
    # print("dfg")