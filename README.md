# pytorch-denoising
This projects compares three architectures of neural networks in the image denoising task. We use simple autoencoder, U-net and U-net based on resnet34 implemented using fastai.

1. Install CUDA drivers for your NVidia graphics card

If you have got two graphics cards use prime select to switch between cards (reboot required):

$ sudo prime-select nvidia

2. Install Conda environment

https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

3. Create the new environment called pytorch-gpu:

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

$ cd

$ source .bashrcConda

$ conda create --name pytorch-gpu

$ conda activate pytorch-gpu

Install all requirements:

$ conda update conda

$ conda install pytorch torchvision cudatoolkit -c pytorch

$ conda install -c pytorch -c fastai fastai

$ conda install -c conda-forge timm

4. Install Pycharm IDE:

https://www.jetbrains.com/help/pycharm/installation-guide.html

Configure your conda environment in pycharm:

https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html

5. Clone this repository

$ git clone https://github.com/dominikbelter/pytorch-denoising

6. Download dataset with example images:

https://www.kaggle.com/jessicali9530/stanford-dogs-dataset

Select 100 training images and copy them to the pytorch-denoising/data/train subfolder

Change resolution of the training images:

$ cd data/train

$ for i in $(ls *.jpg); do convert -resize 96x96^ -gravity center -crop 96x96+0+0 $i th-$i; done

Select 10 testing images and copy them to the pytorch-denoising/data/valid subfolder and change their resolution.

7. Download unet implementation in pytorch:

$ cd models

$ git clone https://github.com/jvanvugt/pytorch-unet

$ mv pytorch-unet pytorchunet

$ touch pytorch-unet/\_\_init\_\_.py

$ cd ..

8. Open the project in Pycharm

change 22 line in the Train.py file to select the architecture

9. Run the main.py script and wait for the results. The images obtained from the neural network are in the main project folder. Play with the parameters in the Params.py if needed.

Example results on the testing data:

autoencoder: Average loss: 0.0043

unet: Average loss: 0.0002

res34 unet: Average loss: 0.0001
