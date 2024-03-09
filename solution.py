
# import libraries
import numpy as np
import random
# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class mlp(nn.Module):
    def __init__(self, time_periods, n_classes, **kwargs):
        super(mlp, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes

        self.dim_in = time_periods * 3
        dim_out = n_classes

        self.n_layers = kwargs.get('n_layers', 3)

        self.fc1 = nn.Linear(in_features=self.dim_in, out_features=kwargs.get('hid_dim_1', 100))
        self.fc2 = nn.Linear(in_features=kwargs.get('hid_dim_1', 100), out_features=kwargs.get('hid_dim_2', 100))
        self.fc3 = nn.Linear(in_features=kwargs.get('hid_dim_2', 100), out_features=kwargs.get('hid_dim_3', 100))

        self.fc4 = nn.Linear(kwargs.get('hid_dim_3', 100), out_features=dim_out)

    def forward(self, x):
        _x = x.reshape(x.shape[0], self.time_periods * 3)

        _x = nn.functional.relu(self.fc1(_x))
        _x = nn.functional.relu(self.fc2(_x))
        _x = nn.functional.relu(self.fc3(_x))

        _x = self.fc4(_x)

        return nn.functional.log_softmax(_x, 1)


class cnn(nn.Module):
    def __init__(self, time_periods, n_sensors, n_classes, **kwargs):
        super(cnn, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=100, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=10)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=160, kernel_size=10)
        self.conv4 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=10)

        self.activation = nn.functional.relu_

        # Pooling and dropout
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)
        #self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.avgpool1 = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(.5)
        # Adaptive pool layer to adjust the size before sending to fully connected layer

        # Fully connected layer
        self.fc = nn.Linear(in_features=160, out_features=6)

    def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, time_periods)
        #_x = x.view(x.shape[0], self.n_sensors, self.time_periods)
        #_x = x.reshape(x.shape[0], self.n_sensors, self.time_periods)
        #_x = x.permute(0, 2, 1)
        # Convolutional layers with ReLU activations
        _x = self.activation(self.conv1(x))
        _x = self.activation(self.conv2(_x))
        _x = self.maxpool1(_x)
        _x = self.activation(self.conv3(_x))
        _x = self.activation(self.conv4(_x))

        # Global average pooling and dropout
        _x = self.avgpool1(_x)
        _x = self.dropout(_x)

        # Flatten the tensor for the fully connected layer
        _x = _x.flatten(start_dim=1)
        # Output layer with softmax activation
        _x = self.fc(_x)

        #pred = nn.functional.log_softmax(_x, 1)

        # output the loss, Use log_softmax for numerical stability
        return nn.functional.log_softmax(_x, 1)


'''
class mlp(object):

  def __init__(self,
               time_periods, n_classes):
        super(mlp, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes
        # WRITE CODE HERE

  def forward(self, x):
    # WRITE CODE HERE
    
    return x
  
# # WRITE CODE HERE

class cnn(object):

  def __init__(self, time_periods, n_sensors, n_classes):
        super(cnn, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        # WRITE CODE HERE
        
        

  def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, time_periods)
        # WRITE CODE HERE
        
        # Layers
        # WRITE CODE HERE
        
        return x
'''