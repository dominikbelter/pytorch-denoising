import torch
from torchvision.utils import save_image
# from Dataset import Dataset
from torch import nn
from skimage.util import random_noise
# from Params import Params
import os
import numpy as np


def image_save(img, name):
    '''
    saves the pics under the given path name
    :param img: tensor size -> torch.Size([1, 3, 32, 96])
    :param name: path name
    :return: none
    '''
    img = img.view(3, 3, 300, 400) # -1 -> when we dont know about size of width/height
    save_image(img, name)


# test
class Test(Params):
    def __init__(self, device, test_folder_input, test_folder_output, results_folder):
        super().__init__()
        self._device = device
        self._criterion_test = nn.MSELoss()
        model = torch.load(self.model_save_PATH)
        model.eval()
        for img_no, (data_in, out_ref) in enumerate( zip(Dataset.test_loader(test_folder_input),Dataset.test_loader(test_folder_output))):
            iter = 0
            for img in data_in:
                # prepare test dataset
                img_test = img.to(device)
                img_out_ref = out_ref[0].to(device)

                test_output = model(img_test[None, ...])
                _, predicted = torch.max(test_output.data, 1)

                output_img = test_output[0].cpu().detach().numpy()
                
                bugs_no_img = np.sum(output_img[1:300, 1:400])
                print("sum: " + str(bugs_no_img*255.0))
                print("bugs no img gauss 8 bit image: " + str(bugs_no_img/(81.0*3.0)/255.0))

                # save_image(img_test, results_folder + f"/{img_no + 1}img_test_in.png")
                # save_image(img_out_ref, results_folder + f"/{img_no + 1}img_test_out.png")
                # save_image(test_output, results_folder + f"/{img_no + 1}img_test_pred.png")
                all_test = torch.cat((img_test, img_out_ref, test_output[0]), 0)
                image_save(all_test, results_folder + f"/{img_no + 1}_all_vs1.png")

                loss_test = + self._criterion_test(test_output[0], img_test).item()
                if iter == self.num_test_images - 1:
                    average_loss_test = loss_test / self.num_test_images
                    print(f'Average loss: {average_loss_test:.4f}')
                    break
                    iter=iter+1