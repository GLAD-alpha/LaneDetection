# %%
# from model import SCNN_Net
import torch
import os
import numpy as np
from loss_function import DiceBCELoss
# %%
USE_DUMMY = True
# %% train data npy to tensor
if not USE_DUMMY : 
    data_dir = '../data/train/data'
    data_npy_list = os.listdir(data_dir)

    train_X = list()
    for itr,npy_files in enumerate(data_npy_list,1):
        print('\r', f"processed : {itr}", end='')
        data = np.load(os.path.join(data_dir, npy_files))
        data = data/255.0
        data = np.transpose(data, (0,3,1,2))
        data = torch.Tensor(data)
        data = torch.unsqueeze(data,0)
        train_X.append(data)

    train_X = torch.cat(train_X,0)
    print('\n',train_X.shape)

    label_dir = '../data/train/label'
    label_npy_list = os.listdir(label_dir)

    train_Y = list()
    for itr,npy_files in enumerate(label_npy_list,1):
        print('\r', f"processed : {itr}", end='')
        data = np.load(os.path.join(label_dir, npy_files))
        data = np.transpose(data, (2,0,1))
        data = torch.FloatTensor(data)
        data = torch.unsqueeze(data,0)
        train_Y.append(data)
    train_Y = torch.cat(train_Y,0)
    print('\n',train_Y.shape)

if USE_DUMMY :
   train_X = torch.randn(100,5,3,720,1280)
   train_Y = torch.randn(100,1,720,1280)
# %% get loader
train_set = torch.utils.data.TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
# %% define model and loss function
from model import LaneNet
model = LaneNet()
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

if USE_CUDA: model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

criterion = DiceBCELoss()
sigmoid = torch.nn.Sigmoid()
# %%
EPOCH = 500
for epoch in range(EPOCH):
  print(f"---------------Epoch : {epoch+1}/{EPOCH}--------------------")
  train_loss = 0.0

  for train_idx, data in enumerate(train_loader, 0):
    optimizer.zero_grad()
    # print('\r',f"training {train_idx+1}/{len(train_loader)}, train_loss: {train_loss:0.4f}",end=" ")
    inputs, labels = data

    for idx in range(len(inputs)):
       model_input = inputs[idx]
       model_gt = labels[idx]
       
       
       print(model_input.shape)
       outputs = model(model_input.to(device))
       assert False

    loss = criterion(outputs, labels.to(device))
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  print('')
# %%
