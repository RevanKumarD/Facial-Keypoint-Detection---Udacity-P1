## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        #----------------------------------------------------------------------------------------------------
        # 1 input image channel (grayscale), 64 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W+2P-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (64, 220, 220)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # after one pool layer the previous output becomes (64, 110, 110)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # second conv layer: 64 inputs, 32 outputs, 5x5 conv
        ## output size = (W+2P-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (32, 106, 106)
        # after another pool layer this becomes (32, 53, 53)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5)
        
        # third conv layer: 32 inputs, 16 outputs, 3x3 conv
        ## output size = (W+2P-F)/S +1 = (53-3)/1 +1 = 51
        # the output tensor will have dimensions: (16, 51, 51)
        # after another pool layer this becomes (16, 25, 25)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3)
        
        #------------------------------------------------------------------------------------------
        # fourth conv layer: 384 inputs, 256 outputs, 3x3 conv
        ## output size = (W+2P-F)/S +1 = (22-3)/1 +1 = 20
        # the output tensor will have dimensions: (256, 20, 20)
        # self.conv4 = nn.Conv2d(384, 256, kernel_size=3)
        
        # fifth conv layer: 256 inputs, 64 outputs, 3x3 conv
        ## output size = (W+2P-F)/S +1 = (20-3)/1 +1 = 18
        # the output tensor will have dimensions: (64, 18, 18)
        # after another pool layer this becomes (64, 9, 9)
        # self.conv5 = nn.Conv2d(256, 64, kernel_size=3)
        
        # maxpool layer
        # after one pool layer the previous output becomes (64, 6, 6)
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        #------------------------------------------------------------------------------------------
        
        # dropout
        self.drop = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(16 * 25 * 25, 1024)
        
        # dropout
        #self.fc1_drop = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(1024, 512)
        
        # 136 output channels (for the 136 keypoints)
        self.fc3 = nn.Linear(512, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # three conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = F.relu(self.conv4(x))
        #x = self.pool(F.relu(self.conv5(x)))
        # x = self.avgpool(x)
        # x = self.drop(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
