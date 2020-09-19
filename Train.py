import torch
from Dataset import Dataset
from torch import nn
from skimage.util import random_noise
from models.Autoencoder import Autoencoder
from models.pytorchunet.unet import UNet
from Params import Params
from fastai.vision.all import *
from timm import create_model

class Train(Params):
    def __init__(self, device):
        super().__init__()

        model_type = 'autoencoder' # change this if you'd like to test various architectures
        if model_type == 'autoencoder':
            model = Autoencoder().cuda()
        if model_type == 'unet':
            model = UNet(n_classes=3, in_channels=3, depth=5, padding=True, up_mode='upconv').cuda()
        if model_type == 'resunet':
            path_dir = './data'
            path = Path(path_dir)
            fnames = get_image_files(path_dir)
            dls = ImageDataLoaders.from_folder(path,valid_pct=0.0)
            m = resnet34()
            m = nn.Sequential(*list(m.children())[:-2])
            model = DynamicUnet(m, 3, (96, 96), norm_type=None).cuda()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        train_load = Dataset.train_loader()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(train_load):
                clean_img_train, _ = data[0], data[1]
                noised_img_train = torch.tensor(
                    random_noise(clean_img_train, mode='s&p', salt_vs_pepper=0.5, clip=True))
                clean_img_train, noised_img_train = clean_img_train.to(device), noised_img_train.to(device)
                output = model(noised_img_train[None, ...])
                loss = criterion(output, clean_img_train[None, ...])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i == self.num_train_images_in_epoch:
                    break
            print(f'epoch [{epoch + 1}/{self.num_epochs}], loss:{loss.item():.4f}')

        torch.save(model, self.model_save_PATH)