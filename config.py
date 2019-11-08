
train_json = 'test_data/my_trainmpii.json'
val_json = 'test_data/my_valmpii.json'
image_root = '/home/mllabs/NewDisk/yww/VOC2000_LYC_20181025_16/train2017'
checkpoint= 'model/'
local_grid_size = (9,9)
batchsize = 16
parts_scale = 0.2


#[hype ]
seed = 42
insize = (384,384)
lreaning_rate=0.002
momentum = 0.9
weight_decay = 0.0005
epochs = 1000
val_epochs = 5
num_workers = 4

lambda_resp = 0.25
lambda_iou = 1.0
lambda_coor = 5.0
lambda_size = 5.0
lambda_limb = 0.5