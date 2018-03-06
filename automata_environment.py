# Convolutional Neural Automata
# Neural networks trained as cellular automata.
# Copyright (C) 2018  Kirby Banman (kirby.banman@gmail.com)

# Redistributable and modifiable under the terms of GNU GPLv3.
# See LICENSE.txt or <http://www.gnu.org/licenses/>


import numpy as np

class AutomataEnvironment:
    
    def __init__(self, rules, grid_height, grid_width):
        self.rules = rules

        self.grid_height = grid_height
        self.grid_width = grid_width
        
        self.current_grid = np.empty((grid_height, grid_width))
        self.previous_grid = np.empty((grid_height, grid_width))

    def copy_state(self, source_environment):
        self.grid_height = source_environment.grid_height
        self.grid_width = source_environment.grid_width

        self.current_grid = np.copy(source_environment.current_grid)
        self.previous_grid = np.copy(source_environment.previous_grid)
        
    def set_state_from_strings(self, string_array, dead_cell = "_", live_cell = "X"):
        self.grid_height = len(string_array)
        self.grid_width = len(string_array[0])
        
        serialized_integer_array = ",".join("".join(string_array)).replace(dead_cell, "0").replace(live_cell, "1")
        self.current_grid = np.fromstring(serialized_integer_array, dtype=float, sep=",").reshape(self.grid_height, self.grid_width)
        self.previous_grid = np.empty((self.grid_height, self.grid_width))
        
    def randomize_state(self, live_probability = 0.2):
        self.current_grid = np.random.choice(
            [1.0, 0.0],
            self.grid_height * self.grid_width,
            p=[live_probability, 1 - live_probability]
        ).reshape(self.grid_height, self.grid_width)

    def iterate(self):
        # Triangle swap the grid references so that current grid is now the previous grid.
        tmp_grid = self.previous_grid
        self.previous_grid = self.current_grid
        self.current_grid = tmp_grid

        # The entire grids are passed to avoid artificially constraining ruleset implementation.
        # EX: - 3x3 von Neumann neighborhood, 5x5 Moore neighborhood, and beyond are possible
        #     - fast implementation via numpy backend or tensor-on-gpu convolution are possible
        self.rules.mutate_current_grid(self.previous_grid, self.current_grid)