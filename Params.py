class Params:
    '''
    Parent class for Test class nad Train class
    '''
    def __init__(self):
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.num_train_images_in_epoch = 152
        self.num_test_images = 10
        self.model_save_PATH = "./denoised_model.pth"

