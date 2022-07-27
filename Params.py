class Params:
    '''
    Parent class for Test class nad Train class
    '''
    def __init__(self):
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 10
        self.num_train_images_in_epoch = 10717
        self.num_test_images = 21
        self.model_save_PATH = "./count_worms_sum.pth"
        self.channels_in = 3
        self.channels_out = 3
        self.img_width = 400
        self.img_height = 300
