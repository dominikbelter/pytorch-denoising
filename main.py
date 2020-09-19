
import torch
from Test import Test
from Train import Train

# Get the name of the current GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Is PyTorch using a GPU?
if torch.cuda.is_available():
    print("CUDA detected");
else:
    print("CUDA not detected");

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = Train(DEVICE)

test = Test(DEVICE)