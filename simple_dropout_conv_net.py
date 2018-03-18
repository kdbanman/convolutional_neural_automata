# Convolutional Neural Automata
# Neural networks trained as cellular automata.
# Copyright (C) 2018  Kirby Banman (kirby.banman@gmail.com)

# Redistributable and modifiable under the terms of GNU GPLv3.
# See LICENSE.txt or <http://www.gnu.org/licenses/>


import torch
import torch.nn as nn
import torch.nn.functional as F

from toroidal_grid_padder import ToroidalGridPadder

class SimpleDropoutConvNet(nn.Module):
    def __init__(self, hidden_channels = 30, dropout_fraction = 0.2):
        super(SimpleDropoutConvNet, self).__init__()

        self.toroidal_padder = ToroidalGridPadder(2)

        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3)
        self.conv1_drop = nn.Dropout2d(p = dropout_fraction)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=3)

    def forward(self, environment_grid):
        convolution_ready_environment_grid = self.toroidal_padder.pad_grid(environment_grid)

        hidden_activations = F.relu(self.conv1(convolution_ready_environment_grid))
        dropout_adjusted_activations = self.conv1_drop(hidden_activations)
        output_activations = F.relu(self.conv2(dropout_adjusted_activations))

        return output_activations

    