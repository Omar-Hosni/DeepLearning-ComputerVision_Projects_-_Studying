from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preprocessing import prepare
from utilities import train


data_dir = 'D:\projects\Tumor Detection\Task03_Liver\Data_Train_Test'
model_dir = 'D:\projects\Tumor Detection\Organ and Tumor Segmentation\\results\\results'
data_in = prepare(data_dir, cache=True)

device = torch.device('cuda')
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16,32,64,128,256),
    strides=(2,2,2,2),
    num_res_units=2,
    norm=Norm.BATCH
).to(device)

loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)