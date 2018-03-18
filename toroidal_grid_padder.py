# Convolutional Neural Automata
# Neural networks trained as cellular automata.
# Copyright (C) 2018  Kirby Banman (kirby.banman@gmail.com)

# Redistributable and modifiable under the terms of GNU GPLv3.
# See LICENSE.txt or <http://www.gnu.org/licenses/>


from enum import Enum

import torch

class ToroidalGridPadder:
    '''
    We don't know the size of the grids this thing will operate at construction time.  But chances are that
    it will be used to pad grids of the same size over and over during it's lifecycle, so cache the scatter
    matrices for reuse.
    '''
    
    def __init__(self, pad_size):
        self._pad_size = pad_size

        self._grid_height = None
        self._grid_width = None
    
    def pad_grid(self, environment_grid):
        if self._unseen_grid_shape(environment_grid.shape):
            self._construct_scatter_indices(environment_grid.shape)

        padded_environment_tensor = torch.nn.ConstantPad2d(self._pad_size, 0)(environment_grid)

        padded_environment_tensor.scatter_(0, self._row_copy_target_index, padded_environment_tensor)
        padded_environment_tensor.scatter_(1, self._col_copy_target_index, padded_environment_tensor)

        # NOTE: I think this .view call breaks batch training, because it assumes a batch size of 1 (first dimension is 1)
        toroidally_wrapped_environment_tensor = padded_environment_tensor.view(1, 1, self._padded_height, self._padded_width).double()

        return toroidally_wrapped_environment_tensor

    def _unseen_grid_shape(self, grid_shape):
        return self._grid_height != grid_shape[0] or self._grid_width != grid_shape[1]

    def _construct_scatter_indices(self, grid_shape):
        self._grid_height = grid_shape[0]
        self._grid_width = grid_shape[1]

        self._padded_height = self._grid_height + self._pad_size * 2
        self._padded_width = self._grid_width + self._pad_size * 2

        self._row_copy_target_index = self._construct_target_scatter_index(self._pad_size, self._padded_height, self._padded_width, Direction.ROWS)
        self._col_copy_target_index = self._construct_target_scatter_index(self._pad_size, self._padded_height, self._padded_width, Direction.COLUMNS)

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
