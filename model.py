# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        
        self.down_conv = nn.Conv2d(48,48,kernel_size=(1,5), padding=(0,2))
        self.up_conv   = nn.Conv2d(48,48,kernel_size=(1,5), padding=(0,2))
        self.right_conv   = nn.Conv2d(48,48,kernel_size=(5,1),padding=(2,0))
        self.left_conv   = nn.Conv2d(48,48,kernel_size=(5,1),padding=(2,0))

    def forward(self,x):
        
        # DOWN CONVOLUTION
        down_slices = list()

        for slice_idx in range(x.shape[2]-1):

            # initial
            if slice_idx==0:
                first_slice = x[:,:,slice_idx,:]
                first_slice = torch.unsqueeze(first_slice, 2)
                down_slices.append(first_slice)
                processed_slice = self.down_conv(first_slice)
            else:
                prior_slice = down_slices[slice_idx]
                processed_slice = self.down_conv(prior_slice)

            next_slice = x[:,:,slice_idx+1,:]
            next_slice = torch.unsqueeze(next_slice,2)
            next_slice = next_slice + processed_slice
            down_slices.append(next_slice)

        down_slices = torch.cat(down_slices,2)

        # print("after down-conv : ", x.shape)
    
        # UP CONVOLUTION
        up_slices = list()

        for slice_idx in range(down_slices.shape[2]-1):
    
            # initial
            if slice_idx==0:
                first_slice = down_slices[:,:,down_slices.shape[2]-1-slice_idx,:]
                first_slice = torch.unsqueeze(first_slice, 2)
                up_slices.append(first_slice)
                processed_slice = self.down_conv(first_slice)
            else:
                prior_slice = up_slices[slice_idx]
                processed_slice = self.down_conv(prior_slice)
            next_slice = down_slices[:,:,down_slices.shape[2]-1-slice_idx-2,:]
            next_slice = torch.unsqueeze(next_slice,2)
            next_slice = next_slice + processed_slice
            up_slices.append(next_slice)

        # reverse order
        up_slices = up_slices[::-1]
        up_slices = torch.cat(up_slices,2)

        # print("after up-conv : ", up_slices.shape)

        # RIGHT CONVOLUTION
        right_slices = list()

        for slice_idx in range(up_slices.shape[3]-1):
            
            if slice_idx==0:
                first_slice = up_slices[:,:,:,slice_idx]
                first_slice = torch.unsqueeze(first_slice,3)
                right_slices.append(first_slice)
                processed_slice = self.right_conv(first_slice)
            else:
                prior_slice = right_slices[slice_idx]
                processed_slice = self.right_conv(prior_slice)

            next_slice = up_slices[:,:,:,slice_idx+1]
            next_slice = torch.unsqueeze(next_slice,3)
            next_slice = next_slice + processed_slice
            right_slices.append(next_slice)
        
        right_slices = torch.cat(right_slices,3)

        # print("after right-conv : ", right_slices.shape)

        # LEFT CONVOLUTION
        left_slices = list()

        for slice_idx in range(right_slices.shape[3]-1):
            
            if slice_idx==0:
                first_slice = right_slices[:,:,:,right_slices.shape[3]-1-slice_idx]
                first_slice = torch.unsqueeze(first_slice,3)
                left_slices.append(first_slice)
                processed_slice = self.left_conv(first_slice)
            else:
                prior_slice = left_slices[slice_idx]
                processed_slice = self.left_conv(prior_slice)

            next_slice = right_slices[:,:,:,right_slices.shape[3]-2-slice_idx]
            next_slice = torch.unsqueeze(next_slice,3)
            next_slice = next_slice + processed_slice
            left_slices.append(next_slice)
        
        left_slices = left_slices[::-1]
        left_slices = torch.cat(left_slices,3)

        # print("after left-conv : ", left_slices.shape)
        return x


'''
input : 5, 720, 1280, 3

output 720, 1280, 1
'''
class SCNN_Net(nn.Module):
    def __init__(self):
        super(SCNN_Net, self).__init__()
        
        self.front_convolution_layer = nn.Conv2d(3,48,10,5,padding=3)
        self.scnn_layer = SCNN()
        self.back_convolution_layer = nn.ConvTranspose2d(48,1,5,5)

    def forward(self,x):
        # print("input shape : ", x.shape)
        x = self.front_convolution_layer(x)
        # print("after first conv : ",x.shape)
        x = self.scnn_layer(x)
        # print("after scnn : ",x.shape)
        x = self.back_convolution_layer(x)
        # print("after last conv : ",x.shape)
        return x
    
class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        
        self.Wxi = nn.Conv2d(24,24,4,2)
        self.Whi = nn.Conv2d(24,24,3,1, padding=1)
        self.Wci = torch.nn.Parameter(torch.randn(24, 43, 78))
        self.Wci.requires_grad = True

        self.Wxf = nn.Conv2d(24,24,4,2)
        self.Whf = nn.Conv2d(24,24,3,1, padding=1) 
        self.Wcf = torch.nn.Parameter(torch.randn(24, 43, 78))
        self.Wcf.requires_grad = True

        self.Wxc = nn.Conv2d(24,24,4,2)
        self.Whc = nn.Conv2d(24,24,3,1, padding=1) 

        self.Wxo = nn.Conv2d(24,24,4,2)
        self.Who = nn.Conv2d(24,24,3,1, padding=1)
        self.Wco =torch.nn.Parameter(torch.randn(24, 43, 78))
        self.Wco.requires_grad = True

        self.input_signal = None
        self.forget_signal = None
        self.cell_signal = None
        self.out_signal = None
        self.hidden_signal = None

    def forward(self, x, previous_cell_signal=None, previous_hidden_signal=None):
        
        if (previous_cell_signal == None) and (previous_hidden_signal == None):
            self.input_signal = torch.sigmoid(self.Wxi(x))
            self.cell_signal = self.input_signal * torch.tanh(self.Wxc(x)) 
            self.out_signal = torch.sigmoid(self.Wxo(x) + self.Wco * self.cell_signal)
            self.hidden_signal = self.out_signal * torch.tanh(self.cell_signal)
        else:
            self.input_signal = torch.sigmoid(self.Wxi(x) + self.Whi(previous_hidden_signal) + self.Wci * previous_cell_signal)
            self.forget_signal = torch.sigmoid(self.Wxf(x) + self.Whf(self.hidden_signal) + self.Wcf * self.cell_signal)
            self.cell_signal = self.forget_signal * self.cell_signal + self.input_signal * torch.tanh(self.Wxc(x) + self.Whc(self.hidden_signal))
            self.out_signal = torch.sigmoid(self.Wxo(x) + self.Who(previous_hidden_signal) + self.Wco * self.cell_signal)
            self.hidden_signal = self.out_signal * torch.tanh(self.cell_signal)

        # print("input_signal shape : ",self.input_signal.shape)
        # print("cell_signal shape : ",self.cell_signal.shape)
        # print("out_signal shape : ",self.out_signal.shape)

        return self.cell_signal, self.hidden_signal

class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()

        self.encoder_1 = nn.Conv2d(3,6,3,2)
        self.encoder_2 = nn.Conv2d(6,12,3,2)
        self.encoder_3 = nn.Conv2d(12,24,3,2)

        self.conv_lstm_1 = ConvLSTM()
        self.conv_lstm_2 = ConvLSTM()
        self.conv_lstm_3 = ConvLSTM()
        
        self.decoder_1_1 = nn.Conv2d(24, 12, 2, 1, padding= 1)
        self.decoder_1_2 = nn.ConvTranspose2d(12,12,3,2)

        self.decoder_2_1 = nn.Conv2d(12, 6, 3, 1, padding= 1)
        self.decoder_2_2 = nn.ConvTranspose2d(6,6,3,2)

        self.decoder_3_1 = nn.Conv2d(6, 3, 3, 1, padding= 1)
        self.decoder_3_2 = nn.ConvTranspose2d(3,3,3,2)

        self.decoder_4_1 = nn.Conv2d(3, 1, 2, 1, padding= 1)
        self.decoder_4_2 = nn.ConvTranspose2d(1,1,2,2)

    def forward(self, x):

        lstm_list = [self.conv_lstm_1, self.conv_lstm_2, self.conv_lstm_3]

        input_1 = self.encoder_3(self.encoder_2(self.encoder_1(x[0]))) # (24, 43, 78)
        input_2 = self.encoder_3(self.encoder_2(self.encoder_1(x[1]))) # (24, 43, 78)
        input_3 = self.encoder_3(self.encoder_2(self.encoder_1(x[2]))) # (24, 43, 78)
        input_4 = self.encoder_3(self.encoder_2(self.encoder_1(x[3]))) # (24, 43, 78)
        input_5 = self.encoder_3(self.encoder_2(self.encoder_1(x[4]))) # (24, 43, 78)

        print("encoder output : ", input_1 .shape)

        for lstm in lstm_list:
            cell_signal_1, hidden_signal_1 = lstm(input_1)
            cell_signal_2, hidden_signal_2 = lstm(input_2, cell_signal_1, hidden_signal_1)
            cell_signal_3, hidden_signal_3 = lstm(input_3, cell_signal_2, hidden_signal_2)
            cell_signal_4, hidden_signal_4 = lstm(input_4, cell_signal_3, hidden_signal_3)
            _            , hidden_signal_5 = lstm(input_5, cell_signal_4, hidden_signal_4)


        print("lstm output : ", hidden_signal_5.shape)

        output = self.decoder_1_1(hidden_signal_5)
        output = self.decoder_1_2(output)
        print("decoder 1 output : ", output.shape)

        output = self.decoder_2_1(output)
        output = self.decoder_2_2(output)
        print("decoder 2 output : ", output.shape)

        output = self.decoder_3_1(output)
        output = self.decoder_3_2(output)

        print("decoder 3 output : ", output.shape)

        output = self.decoder_4_1(output)
        output = self.decoder_4_2(output)

        print("decoder 4 output : ", output.shape)
        return x