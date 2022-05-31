
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

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
 
    def forward(self, inputs, targets, smooth=1):        
        
        inputs = F.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class LossMasked(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LossMasked, self).__init__()
 
    def forward(self, inputs, targets, mask, reduction='mean'):
        flat_mask = torch.nn.Flatten()(mask)
        count = torch.count_nonzero(flat_mask, 1)

        if reduction == 'mean':
            return torch.sum(torch.abs((inputs - targets) * mask)) / torch.sum(count)
        elif reduction == 'sum':
            return torch.sum(torch.abs((inputs - targets) * mask))
        else:
            return torch.sum(torch.abs((inputs - targets) * mask)) / torch.sum(count)


class Train(Params):
    def __init__(self, device, dataset_folder, train_folder_input, train_folder_output, results_folder):
        super().__init__()

        model_type = 'resunet' # change this if you'd like to test various architectures
        if model_type == 'autoencoder':
            model = Autoencoder().cuda()
        if model_type == 'unet':
            model = UNet(n_classes=1, in_channels=self.channels_in, depth=5, padding=True, up_mode='upconv').cuda()
        if model_type == 'resunet':
            # path_dir = dataset_folder
            # path = Path(path_dir)
            # fnames = get_image_files(path_dir)
            # dls = ImageDataLoaders.from_folder(path,valid_pct=0.0)
            m = resnet18()
            m = nn.Sequential(*list(m.children())[:-2])
            model = DynamicUnet(m, self.channels_out, (400, 300), norm_type=None).cuda()

        criterion_MAE = nn.L1Loss()
        criterion_dice = DiceLoss(nn)
        criterion_masked = LossMasked(nn)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        #train_load_input = #project: separate folder for input
        #train_load_output = #project: separate folder for output
        for epoch in range(self.num_epochs):
            for i, (data_in, data_out) in enumerate(zip(Dataset.train_loader(train_folder_input), Dataset.train_loader(train_folder_output))): #project: iterate over in and out folders
                in_img_train, _ = data_in, data_in[1]
                out_img_train, _ = data_out, data_out[1] #project: prepare output data
                # save_image(data_in, results_folder + "/img_data_in.png")
                # save_image(in_img_train[0], results_folder + "/img_data_in0.png")
                # save_image(in_img_train[1], results_folder + "/img_data_in1.png")
                # save_image(in_img_train[2], results_folder + "/img_data_in2.png")
                
                in_img_train, out_img_train = in_img_train.to(device), out_img_train.to(device)
                output = model(in_img_train) #project: be careful about input/output images
                if epoch == 9:
                    save_image(output, results_folder + "/img_data_out.png")
                    save_image(out_img_train, results_folder + "/img_data_out_ref.png")
                    save_image(in_img_train, results_folder + "/img_data_in.png")
                loss_MAE = criterion_MAE(output, out_img_train)
                #loss_masked = criterion_masked(output,out_img_train, out_img_train)
                loss_dice = criterion_dice(output, out_img_train)
                loss = 32*loss_dice + loss_MAE
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i == self.num_train_images_in_epoch:
                    break
            print(f'epoch [{epoch + 1}/{self.num_epochs}], loss:{loss.item():.4f}')

        torch.save(model, self.model_save_PATH)
