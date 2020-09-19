import torch
from torchvision.utils import save_image
from Dataset import Dataset
from torch import nn
from skimage.util import random_noise
from Params import Params
import os


def image_save(img, name):
    '''
    saves the pics under the given path name
    :param img: tensor size -> torch.Size([1, 3, 32, 96])
    :param name: path name
    :return: none
    '''
    img = img.view(3, 3, 96, -1) # -1 -> when we dont know about size of width/height
    save_image(img, name)


# test
class Test(Params):
    def __init__(self, device):
        super().__init__()
        self._device = device
        self._criterion_test = nn.MSELoss()
        model = torch.load(self.model_save_PATH)
        model.eval()
        for i, data in enumerate(Dataset.test_loader()):
            iter = 0
            for img in data:
                # prepare test dataset
                clean_img_test = img
                noised_img_test = torch.tensor(random_noise(clean_img_test, mode='s&p', salt_vs_pepper=0.5, clip=True))
                clean_img_test, noised_img_test = clean_img_test.to(device), noised_img_test.to(device)

                test_output = model(noised_img_test[None, ...])
                _, predicted = torch.max(test_output.data, 1)

                all_test = torch.cat((clean_img_test, noised_img_test, test_output[0]), 0)
                image_save(all_test, f"./{iter + 1}_all_vs1.png")

                loss_test = + self._criterion_test(test_output[0], clean_img_test).item()
                if iter == self.num_test_images - 1:
                    average_loss_test = loss_test / self.num_test_images
                    print(f'Average loss: {average_loss_test:.4f}')
                    break
                iter=iter+1
