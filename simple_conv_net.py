from enum import Enum


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    def __init__(self, hidden_channels = 10):
        super(SimpleConvNet, self).__init__()

        self.toroidal_boundary_padding_size = 2

        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=3)

    def forward(self, environment_grid):
        '''
        This method doesn't look like a standard convolutional forward pass, but it needs
        to do two special things: return an output tensor of the same dimension as the
        input, and treat the tensor boundaries as toroidally wrapping.
        
        NOTES:
        - environment_grid must be a 4D Tensor, not a 2D numpy array.
        - two 3x3 kernels means 2 layers will be taken from all sides, so toroidally pad
          the sides with copies of the unpadded boundaries.
        '''
        pad_size = self.toroidal_boundary_padding_size
        convolution_ready_environment_grid = self._prepare_environment_grid_for_forward_pass(environment_grid, pad_size)

        hidden_activations = F.relu(self.conv1(convolution_ready_environment_grid))
        return F.relu(self.conv2(hidden_activations))

    def _prepare_environment_grid_for_forward_pass(self, environment_grid, pad_size):
        grid_height = environment_grid.shape[0]
        grid_width = environment_grid.shape[1]

        padded_height = grid_height + pad_size * 2
        padded_width = grid_width + pad_size * 2

        padded_environment_tensor = torch.nn.ConstantPad2d(pad_size, 0)(environment_grid)

        row_copy_target_index = self._construct_target_scatter_index(pad_size, padded_height, padded_width, Direction.ROWS)
        col_copy_target_index = self._construct_target_scatter_index(pad_size, padded_height, padded_width, Direction.COLUMNS)

        padded_environment_tensor.scatter_(0, row_copy_target_index, padded_environment_tensor)
        padded_environment_tensor.scatter_(1, col_copy_target_index, padded_environment_tensor)

        # NOTE: I think this .view call breaks batch training, because it assumes a batch size of 1 (first dimension is 1)
        convolution_ready_environment_tensor = padded_environment_tensor.view(1, 1, padded_height, padded_width).double()

        return convolution_ready_environment_tensor

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

class Direction(Enum):
    ROWS = 1
    COLUMNS = 2
