from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        ) # output_size = 24

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 9


        # CONVOLUTION BLOCK 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 5

        # OUTPUT BLOCK
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 5
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 3
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.dropout(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Net1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 28

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 28
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 26

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 13
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 13

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 11
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 11

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 9
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
        ) # output_size = 7
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
        ) # output_size = 5
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        #x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        #x = self.dropout(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 28

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 28
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) # output_size = 26

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2) # output_size = 13
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 13

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 11
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 11

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 9
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 5

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 3

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        #x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        #x = self.dropout(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class NetBatchNormalization(nn.Module):
    def __init__(self):
        super(NetBatchNormalization, self).__init__()
        # Input Block C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 30

        # CONVOLUTION BLOCK
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 28

        # CONVOLUTION BLOCK
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 28

         # TRANSITION BLOCK
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 28-3/3 +1 = 14

        # CONVOLUTION BLOCK
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 12

        # CONVOLUTION BLOCK
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 8


        # CONVOLUTION BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 8

        # TRANSITION BLOCK
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 8-2/2 +1 = 4

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 2

        # OUTPUT BLOCK
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 1
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        ) # output_size = 1
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.convblock1(x) # output_size = 30
        x = self.convblock2(x) # output_size = 28
        x = self.convblock3(x) # output_size = 30 -- 1*1
        x = self.pool1(x) # output_size = 15
        x = self.convblock4(x) # output_size = 15 13
        x = self.convblock5(x) # output_size = 13 11
        x = self.convblock6(x) # output_size = 11  9
        x = self.convblock7(x) # output_size = 11 -- 1*1
        x = self.pool1(x) # output_size = 5
        x = self.convblock8(x) # output_size = 3
        x = self.convblock9(x) # output_size = 3
        x = self.convblock10(x) # output_size = 3
        x = self.gap(x) # output_size = 1
        x = self.convblock11(x) # output_size = 1 -- 1*1
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class NetLayerNormalization(nn.Module):
    def __init__(self):
        input_shape=[3,32, 32] 
        super(NetLayerNormalization, self).__init__()
        # Input Block C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            #nn.BatchNorm2d(16),
            nn.LayerNorm((16, 30, 30), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 30

        # CONVOLUTION BLOCK
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((16, 28, 28), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 28

        # CONVOLUTION BLOCK
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            nn.LayerNorm((16, 30, 30), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 28

         # TRANSITION BLOCK
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 28-3/3 +1 = 14

        # CONVOLUTION BLOCK
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((16, 13, 13), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 12

        # CONVOLUTION BLOCK
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((16, 11, 11), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((32, 9, 9), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 8


        # CONVOLUTION BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
            nn.LayerNorm((32, 11, 11), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 8

        # TRANSITION BLOCK
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 8-2/2 +1 = 4

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((32, 3, 3), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 2

        # OUTPUT BLOCK
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm((32, 3, 3), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 1
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        ) # output_size = 1
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.convblock1(x) # output_size = 30
        x = self.convblock2(x) # output_size = 28
        x = self.convblock3(x) # output_size = 30 -- 1*1
        x = self.pool1(x) # output_size = 15
        x = self.convblock4(x) # output_size = 15 13
        x = self.convblock5(x) # output_size = 13 11
        x = self.convblock6(x) # output_size = 11  9
        x = self.convblock7(x) # output_size = 11 -- 1*1
        x = self.pool1(x) # output_size = 5
        x = self.convblock8(x) # output_size = 3
        x = self.convblock9(x) # output_size = 3
        x = self.convblock10(x) # output_size = 3
        x = self.gap(x) # output_size = 1
        x = self.convblock11(x) # output_size = 1 -- 1*1
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
class NetGroupNormalization(nn.Module):
    def __init__(self):
        input_shape=[3,32, 32]
        super(NetGroupNormalization, self).__init__()
        # Input Block C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            #nn.BatchNorm2d(16),
            nn.GroupNorm(4, 16),
            nn.ReLU()
        ) # output_size = 30

        # CONVOLUTION BLOCK
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            #nn.LayerNorm((16, 28, 28), elementwise_affine=False),
            nn.ReLU()
        ) # output_size = 28

        # CONVOLUTION BLOCK
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            #nn.LayerNorm((16, 30, 30), elementwise_affine=False),
            nn.GroupNorm(4, 16),
            nn.ReLU()
        ) # output_size = 28

         # TRANSITION BLOCK
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 28-3/3 +1 = 14

        # CONVOLUTION BLOCK
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            #nn.LayerNorm((16, 13, 13), elementwise_affine=False),
            nn.GroupNorm(4, 16),
            nn.ReLU()
        ) # output_size = 12

        # CONVOLUTION BLOCK
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            #nn.LayerNorm((16, 11, 11), elementwise_affine=False),
            nn.GroupNorm(4, 16),
            nn.ReLU()
        ) # output_size = 10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            #nn.LayerNorm((32, 9, 9), elementwise_affine=False),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        ) # output_size = 8


        # CONVOLUTION BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
            #nn.LayerNorm((32, 11, 11), elementwise_affine=False),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        ) # output_size = 8

        # TRANSITION BLOCK
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 8-2/2 +1 = 4

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            #nn.LayerNorm((32, 3, 3), elementwise_affine=False),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        ) # output_size = 2

        # OUTPUT BLOCK
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            #nn.LayerNorm((32, 3, 3), elementwise_affine=False),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        ) # output_size = 1
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        ) # output_size = 1
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1

        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.convblock1(x) # output_size = 30
        x = self.convblock2(x) # output_size = 28
        x = self.convblock3(x) # output_size = 30 -- 1*1
        x = self.pool1(x) # output_size = 15
        x = self.convblock4(x) # output_size = 15 13
        x = self.convblock5(x) # output_size = 13 11
        x = self.convblock6(x) # output_size = 11  9
        x = self.convblock7(x) # output_size = 11 -- 1*1
        x = self.pool1(x) # output_size = 5
        x = self.convblock8(x) # output_size = 3
        x = self.convblock9(x) # output_size = 3
        x = self.convblock10(x) # output_size = 3
        x = self.gap(x) # output_size = 1
        x = self.convblock11(x) # output_size = 1 -- 1*1
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)





