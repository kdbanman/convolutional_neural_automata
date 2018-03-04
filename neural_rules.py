from enum import Enum

import torch
import numpy as np

class Direction(Enum):
    ROWS = 1
    COLUMNS = 2

class NeuralRules:

    def __init__(self, conv_net):
        self.conv_net = conv_net

    def mutate_current_grid(self, previous_grid, current_grid):
        grid_height = current_grid.shape[0]
        grid_width = current_grid.shape[1]

        pad_size = self.conv_net.toroidal_boundary_padding_size
        padded_height = grid_height + pad_size * 2
        padded_width = grid_width + pad_size * 2


        previous_grid_tensor = torch.autograd.Variable(torch.from_numpy(previous_grid))
        padded_previous_grid = torch.nn.ConstantPad2d(pad_size, 0)(previous_grid_tensor)

        row_copy_target_index = self._construct_target_scatter_index(pad_size, padded_height, padded_width, Direction.ROWS)
        col_copy_target_index = self._construct_target_scatter_index(pad_size, padded_height, padded_width, Direction.COLUMNS)

        padded_previous_grid.scatter_(0, row_copy_target_index, padded_previous_grid)
        padded_previous_grid.scatter_(1, col_copy_target_index, padded_previous_grid)

        convolution_ready_previous_grid = padded_previous_grid.view(1, 1, padded_height, padded_width).double()
        computed_grid = self.conv_net(convolution_ready_previous_grid).data.numpy().reshape((grid_height, grid_width))
        
        np.copyto(current_grid, computed_grid)


    def _construct_target_scatter_index(self, pad_size, grid_height, grid_width, direction):
        '''
        EX: To swap rows for pad_size 2, and a 10x10 grid, construct a Tensor.scatter_ index like this:
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, ],  # row 2 gets pushed to row 8 (N - 2)
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, ],  # row 3 gets pushed to row 9 (N - 1)
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, ],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # row 6 (N - 4) gets pushed to row 0
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],  # row 7 (N - 3) gets pushed to row 1
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, ],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, ],
        ]

        The columnar swap version is just the transpose of above.
        '''
        scatter_target_count = None
        if direction == Direction.ROWS:
            scatter_target_count = grid_height
        else:
            scatter_target_count = grid_width

        # Enumerate targets in order in 1D
        copy_target_list = torch.linspace(0, scatter_target_count - 1, steps = scatter_target_count).long()

        # Set new targets to toroidally wrap in 1D
        for i in range(0, pad_size):
            copy_target_list[pad_size + i] = scatter_target_count - pad_size + i
            copy_target_list[scatter_target_count - pad_size * 2 + i] = i

        pre_expansion_view_dimensions = None
        if direction == Direction.ROWS:
            pre_expansion_view_dimensions = (grid_height, 1)
        else:
            pre_expansion_view_dimensions = (1, grid_width)

        # Expand targets to full grid size for scatter call
        return torch.autograd.Variable(copy_target_list).view(pre_expansion_view_dimensions).expand(grid_height, grid_width)