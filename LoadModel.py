
import torch
from Dataset import Dataset
from torch import nn
from skimage.util import random_noise
from models.Autoencoder import Autoencoder
from models.pytorchunet.unet import UNet
from Params import Params
from fastai.vision.all import *
from torchvision.transforms.functional import *
from torchvision.utils import save_image
from timm import create_model

class LoadModel(Params):
    def __init__(self, device):
        super().__init__()
        if str(device) == "cpu":
            print("load cpu")
            model = torch.load(self.model_save_PATH, map_location=torch.device('cpu'))
        else :
            print("load gpu")
            model = torch.load(self.model_save_PATH)