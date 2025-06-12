# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, f, filters, stage, block):
        """
        Arguments:
        in_channels -- number of input channels
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        """
        super(IdentityBlock, self).__init__()
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # First component of main path
        self.conv2a = nn.Conv2d(in_channels, F1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2a = nn.BatchNorm2d(F1)
        
        # Second component of main path
        self.conv2b = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding=f//2, bias=False)
        self.bn2b = nn.BatchNorm2d(F2)
        
        # Third component of main path
        self.conv2c = nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2c = nn.BatchNorm2d(F3)
        
        # Initialize weights with Xavier uniform (equivalent to glorot_uniform)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, X):
        """
        Arguments:
        X -- input tensor of shape (batch_size, n_C_prev, n_H_prev, n_W_prev)
        
        Returns:
        X -- output of the identity block, tensor of shape (batch_size, n_C, n_H, n_W)
        """
        
        # Save the input value for shortcut connection
        X_shortcut = X
        
        # First component of main path
        X = self.conv2a(X)
        X = self.bn2a(X)
        X = F.relu(X)
        
        # Second component of main path
        X = self.conv2b(X)
        X = self.bn2b(X)
        X = F.relu(X)
        
        # Third component of main path
        X = self.conv2c(X)
        X = self.bn2c(X)
        
        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = X + X_shortcut
        X = F.relu(X)
        
        return X


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, f, filters, stage, block, s=2):
        """
        Arguments:
        in_channels -- number of input channels
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used
        """
        super(ConvolutionalBlock, self).__init__()
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        ##### MAIN PATH #####
        # First component of main path
        self.conv2a = nn.Conv2d(in_channels, F1, kernel_size=1, stride=s, padding=0, bias=False)
        self.bn2a = nn.BatchNorm2d(F1)
        
        # Second component of main path
        self.conv2b = nn.Conv2d(F1, F2, kernel_size=f, stride=1, padding=f//2, bias=False)
        self.bn2b = nn.BatchNorm2d(F2)
        
        # Third component of main path
        self.conv2c = nn.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2c = nn.BatchNorm2d(F3)
        
        ##### SHORTCUT PATH ####
        self.conv_shortcut = nn.Conv2d(in_channels, F3, kernel_size=1, stride=s, padding=0, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(F3)
        
        # Initialize weights with Xavier uniform (equivalent to glorot_uniform)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, X):
        """
        Arguments:
        X -- input tensor of shape (batch_size, n_C_prev, n_H_prev, n_W_prev)
        
        Returns:
        X -- output of the convolutional block, tensor of shape (batch_size, n_C, n_H, n_W)
        """
        
        # Save the input value
        X_shortcut = X
        
        ##### MAIN PATH #####
        # First component of main path
        X = self.conv2a(X)
        X = self.bn2a(X)
        X = F.relu(X)
        
        # Second component of main path
        X = self.conv2b(X)
        X = self.bn2b(X)
        X = F.relu(X)
        
        # Third component of main path
        X = self.conv2c(X)
        X = self.bn2c(X)
        
        ##### SHORTCUT PATH ####
        X_shortcut = self.conv_shortcut(X_shortcut)
        X_shortcut = self.bn_shortcut(X_shortcut)
        
        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = X + X_shortcut
        X = F.relu(X)
        
        return X


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, input_shape=(3, 256, 256), n_classes=1):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images (channels, height, width) - PyTorch format
        n_classes -- integer, number of classes
        """
        super(ResNet50, self).__init__()
        
        # Zero-Padding (equivalent to ZeroPadding2D((3, 3)))
        self.pad = nn.ZeroPad2d(3)
        
        # Stage 1
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=1, padding=0, bias=False)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 2
        self.conv_block_2a = ConvolutionalBlock(32, f=3, filters=[32, 32, 128], stage=2, block='a', s=1)
        self.id_block_2b = IdentityBlock(128, f=3, filters=[32, 32, 128], stage=2, block='b')
        self.id_block_2c = IdentityBlock(128, f=3, filters=[32, 32, 128], stage=2, block='c')
        
        # Stage 3
        self.conv_block_3a = ConvolutionalBlock(128, f=3, filters=[64, 64, 256], stage=3, block='a', s=2)
        self.id_block_3b = IdentityBlock(256, f=3, filters=[64, 64, 256], stage=3, block='b')
        self.id_block_3c = IdentityBlock(256, f=3, filters=[64, 64, 256], stage=3, block='c')
        self.id_block_3d = IdentityBlock(256, f=3, filters=[64, 64, 256], stage=3, block='d')
        
        # Stage 4
        self.conv_block_4a = ConvolutionalBlock(256, f=3, filters=[128, 128, 512], stage=4, block='a', s=2)
        self.id_block_4b = IdentityBlock(512, f=3, filters=[128, 128, 512], stage=4, block='b')
        self.id_block_4c = IdentityBlock(512, f=3, filters=[128, 128, 512], stage=4, block='c')
        self.id_block_4d = IdentityBlock(512, f=3, filters=[128, 128, 512], stage=4, block='d')
        self.id_block_4e = IdentityBlock(512, f=3, filters=[128, 128, 512], stage=4, block='e')
        self.id_block_4f = IdentityBlock(512, f=3, filters=[128, 128, 512], stage=4, block='f')
        
        # Stage 5
        self.conv_block_5a = ConvolutionalBlock(512, f=3, filters=[256, 256, 1024], stage=5, block='a', s=2)
        self.id_block_5b = IdentityBlock(1024, f=3, filters=[256, 256, 1024], stage=5, block='b')
        self.id_block_5c = IdentityBlock(1024, f=3, filters=[256, 256, 1024], stage=5, block='c')
        
        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output layer - REMOVED nn.Flatten()
        self.fc = nn.Linear(1024, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Arguments:
        x -- input tensor of shape (batch_size, channels, height, width)
        
        Returns:
        x -- output tensor of shape (batch_size, n_classes)
        """
        
        # Zero-Padding
        x = self.pad(x)
        
        # Stage 1
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # Stage 2
        x = self.conv_block_2a(x)
        x = self.id_block_2b(x)
        x = self.id_block_2c(x)
        
        # Stage 3
        x = self.conv_block_3a(x)
        x = self.id_block_3b(x)
        x = self.id_block_3c(x)
        x = self.id_block_3d(x)
        
        # Stage 4
        x = self.conv_block_4a(x)
        x = self.id_block_4b(x)
        x = self.id_block_4c(x)
        x = self.id_block_4d(x)
        x = self.id_block_4e(x)
        x = self.id_block_4f(x)
        
        # Stage 5
        x = self.conv_block_5a(x)
        x = self.id_block_5b(x)
        x = self.id_block_5c(x)
        
        # Average pooling
        x = self.avgpool(x)
        
        # Flatten using view() instead of nn.Flatten()
        x = x.view(x.size(0), -1)  # This replaces self.flatten(x)
        x = self.fc(x)
        
        # Apply softmax if more than 1 class, sigmoid if binary classification
        if self.fc.out_features > 1:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        
        return x
