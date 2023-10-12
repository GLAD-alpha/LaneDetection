# %%
from model import SCNN_Net
import torch
import os
import numpy as np
from loss_function import DiceBCELoss
# %%
def model_test():
    dummy_input = torch.randn(1,3,720,1280)
    model = SCNN_Net()
    x = model(dummy_input)
    assert x.shape == (1,1,720,1280)

model_test()
# %%
data_dir = '../data/train/data'
data_npy_list = os.listdir(data_dir)

train_X = list()
for itr,npy_files in enumerate(data_npy_list,1):
    print('\r', f"processed : {itr}", end='')
    data = np.load(os.path.join(data_dir, npy_files))
    data = data/255.0
    data = np.transpose(data, (2,0,1))
    data = torch.Tensor(data)
    data = torch.unsqueeze(data,0)
    train_X.append(data)

train_X = torch.cat(train_X,0)
print(train_X.shape)
# %%
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
print(train_Y.shape)
# %%
train_set = torch.utils.data.TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
# %%
model = SCNN_Net()
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
    print('\r',f"training {train_idx+1}/{len(train_loader)}, train_loss: {train_loss:0.4f}",end=" ")
    inputs, labels = data

    outputs = model(sigmoid(inputs.to(device)))
    # print("output : ",outputs.shape)
    # print("\n")
    loss = criterion(outputs, labels.to(device))
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  print('')
# %%
test_input = np.load('/home/data/train/data/clips_0313-1_60_road.npy')
test_input = np.transpose(test_input,(2,0,1))
test_input = np.expand_dims(test_input,0)
test_input = torch.FloatTensor(test_input)

test_input  = test_input.to(device)
# %%
test_label = np.load('/home/data/train/label/clips_0313-1_60_lane.npy')
test_label = np.squeeze(test_label)
# %%
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

with torch.no_grad():

    y_pred = model(test_input)
    y_pred =y_pred.cpu().detach().numpy()
    y_pred = np.squeeze(y_pred)

    x = y_pred.flatten()
    x = np.expand_dims(x,-1)
    y = test_label.flatten()

    log_reg = LogisticRegression().fit(x,y)
# %%
binary_img = log_reg.predict(x)
binary_img = binary_img.reshape((720,1280))
# %%
plt.imshow(binary_img, cmap='gray')