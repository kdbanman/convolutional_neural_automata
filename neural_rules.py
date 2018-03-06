# Convolutional Neural Automata
# Neural networks trained as cellular automata.
# Copyright (C) 2018  Kirby Banman (kirby.banman@gmail.com)

# Redistributable and modifiable under the terms of GNU GPLv3.
# See LICENSE.txt or <http://www.gnu.org/licenses/>


import torch
import numpy as np

class NeuralRules:

    def __init__(self, conv_net, round_to_integers = False):
        self.conv_net = conv_net
        self.round_to_integers = round_to_integers

    def mutate_current_grid(self, previous_grid, current_grid):
        previous_grid_tensor = torch.autograd.Variable(torch.from_numpy(previous_grid))
        computed_tensor = self.conv_net(previous_grid_tensor)

        # Get numpy array back and drop batch and channel placeholder dimensions.
        computed_grid = computed_tensor.data.numpy().reshape(current_grid.shape)
        np.copyto(current_grid, computed_grid)

        if self.round_to_integers:
            np.round(current_grid)