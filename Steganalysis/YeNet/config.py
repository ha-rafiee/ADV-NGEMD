mode = 'train' # train or test

epochs = 100

lr = 2e-4
weight_decay = 1e-5
gamma = 0.5
weight_decay_step = 30

train_batch_size = 4
val_batch_size = 4
test_batch_size = 4
save_freq = 2
val_freq = 2
strat_save_epoch = 2

train_data_dir = ''
val_data_dir = '' 
test_data_dir = '' 
stego_img_height = 512 # stego_img_height == stego_img_width 
stego_img_channel = 3 # 



pre_trained_yenet_path = 'checkpoints/YeNet/checkpoint_100.pt'
