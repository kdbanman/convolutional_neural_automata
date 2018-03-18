# Convolutional Neural Automata
# Neural networks trained as cellular automata.
# Copyright (C) 2018  Kirby Banman (kirby.banman@gmail.com)

# Redistributable and modifiable under the terms of GNU GPLv3.
# See LICENSE.txt or <http://www.gnu.org/licenses/>

# Conway's rules by numpy fft taken from here:
# https://github.com/thearn/game-of-life

import numpy as np
from numpy.fft import fft2, ifft2

class ConwaysRules:

    def fft_convolve2d(self, previous_grid, kernel):
        fourier_transformed = fft2(previous_grid)
        transformed_by_flipped_kernel = fft2(np.flipud(np.fliplr(kernel)))

        transformed_height, transformed_width = fourier_transformed.shape
        inverse_transformed = np.real(ifft2(fourier_transformed*transformed_by_flipped_kernel))

        vertically_translated_inverse = np.roll(inverse_transformed, - int(transformed_height / 2) + 1, axis=0)
        final_inverse = np.roll(vertically_translated_inverse, - int(transformed_width / 2) + 1, axis=1)

        return final_inverse

    def mutate_current_grid(self, previous_grid, current_grid):
        grid_height, grid_width = previous_grid.shape

        small_von_neuman_kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        von_neuman_kernel = np.zeros((grid_height, grid_width))
        von_neuman_kernel[grid_height // 2 - 1 : grid_height // 2+2, grid_width // 2 - 1 : grid_width // 2 + 2] = small_von_neuman_kernel

        neighborhood_sums_grid = self.fft_convolve2d(previous_grid, von_neuman_kernel).round()

        current_grid.fill(0)
        current_grid[np.where((neighborhood_sums_grid == 2) & (previous_grid == 1))] = 1
        current_grid[np.where((neighborhood_sums_grid == 3) & (previous_grid == 1))] = 1

        current_grid[np.where((neighborhood_sums_grid == 3) & (previous_grid == 0))] = 1