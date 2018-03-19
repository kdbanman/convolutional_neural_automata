# Convolutional Neural Automata
# Neural networks trained as cellular automata.
# Copyright (C) 2018  Kirby Banman (kirby.banman@gmail.com)

# Redistributable and modifiable under the terms of GNU GPLv3.
# See LICENSE.txt or <http://www.gnu.org/licenses/>


class ConwaysRules:
    def mutate_current_grid(self, previous_grid, current_grid):
        grid_height = current_grid.shape[0]
        grid_width = current_grid.shape[1]

        # Overwrite the current grid based on the previous grid contents.
        for row in range(grid_height):
            for col in range(grid_width):
                live_neighbours = (
                    previous_grid[row, (col - 1) % grid_width] +
                    previous_grid[row, (col + 1) % grid_width] +
                    previous_grid[(row - 1) % grid_height, col] +
                    previous_grid[(row + 1) % grid_height, col] +
                    previous_grid[(row - 1) % grid_height, (col - 1) % grid_width] +
                    previous_grid[(row - 1) % grid_height, (col + 1) % grid_width] +
                    previous_grid[(row + 1) % grid_height, (col - 1) % grid_width] +
                    previous_grid[(row + 1) % grid_height, (col + 1) % grid_width]
                )
                if previous_grid[row, col] == 1:
                    if live_neighbours == 2 or live_neighbours == 3:
                        current_grid[row, col] = 1
                    else:
                        current_grid[row, col] = 0
                else:
                    if live_neighbours == 3:
                        current_grid[row, col] = 1
                    else:
                        current_grid[row, col] = 0