
import torch
from Test import Test
from Train import Train

torch.cuda.empty_cache()

# Get the name of the current GPU

# Is PyTorch using a GPU?
if torch.cuda.is_available():
    print("CUDA detected")
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA not detected")
    

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_folder = '/home/dominik/Sources/pytorch-denoising/dataRGB/train'
train_folder_input = '/home/dominik/Sources/pytorch-denoising/dataRGB/train/input' #project: separate folder for input
train_folder_output = '/home/dominik/Sources/pytorch-denoising/dataRGB/train/output' #project: separate folder for output

test_folder = '/home/dominik/Sources/pytorch-denoising/dataRGB/test'
test_folder_input = '/home/dominik/Sources/pytorch-denoising/dataRGB/test/input'
test_folder_output = '/home/dominik/Sources/pytorch-denoising/dataRGB/test/output'

results_folder = '/home/dominik/Sources/pytorch-denoising/dataRGB/results'

train = Train(DEVICE, dataset_folder, train_folder_input, train_folder_output)

test = Test(DEVICE, test_folder_input, test_folder_output, results_folder)
