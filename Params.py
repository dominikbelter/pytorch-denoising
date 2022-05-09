class Params:
    '''
    Parent class for Test class nad Train class
    '''
    def __init__(self):
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.num_train_images_in_epoch = 500
        self.num_test_images = 10
        self.model_save_PATH = "./denoised_model.pth"
        self.channels_in = 3
        self.channels_out = 3