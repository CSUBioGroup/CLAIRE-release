class Args(object):
    dname= None
                       
    gpu = '0'
    workers = 6
    seed = 39
    n_repeat = 1

    skip_training = False

    # model arch settings
    lat_dim = 128
    block_level = 1
    moco_k = 2048
    moco_m = 0.999
    moco_t = 0.07
    init = 'uniform'
    
    # training params
    lr = 1e-4
    adjustLr = False
    cos = False
    schedule = []
    optim = 'Adam'
    momentum = 0.9
    weight_decay = 1e-5

    # dataset params
    select_hvg = 2000
    knn = 10
    alpha = .5
    augment_set = ['int']

    anchor_schedule = []
    fltr = 'gmm'
    yita = .5

    symmetric=True

    epochs = 80
    start_epoch = 0
    batch_size = 256

    # logging settings
    print_freq = 10
    save_freq = 10
    visualize_ckpts = [10, 20, 40, 80, 120]

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")