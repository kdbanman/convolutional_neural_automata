# Convolutional Neural Automata
# Neural networks trained as cellular automata.
# Copyright (C) 2018  Kirby Banman (kirby.banman@gmail.com)

# Redistributable and modifiable under the terms of GNU GPLv3.
# See LICENSE.txt or <http://www.gnu.org/licenses/>


import torch
import torch.nn as nn
import torch.nn.functional as F

from toroidal_grid_padder import ToroidalGridPadder

class SimpleConvNet(nn.Module):
    def __init__(self, hidden_channels = 10):
        super(SimpleConvNet, self).__init__()

        self.toroidal_padder = ToroidalGridPadder(2)

        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=3)

    def forward(self, environment_grid):
        '''
        This method differs from a standard convolutional forward pass, because it needs
        to do two special things: return an output tensor of the same dimension as the
        input, and treat the tensor boundaries as toroidally wrapping.
        
        NOTES:
        - environment_grid must be a 4D Tensor, not a 2D numpy array.
        - padding example: two 3x3 kernels means 2 layers will be taken from all sides,
          so pad the sides with copies of the unpadded boundaries. (i.e. toroidal padding)
        '''
        convolution_ready_environment_grid = self.toroidal_padder.pad_grid(environment_grid)

        hidden_activations = F.relu(self.conv1(convolution_ready_environment_grid))
        return F.relu(self.conv2(hidden_activations))

    