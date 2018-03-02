import numpy as np

class ConwaysEnvironment:
    
    def __init__(self, grid_height, grid_width):
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        self.previous_grid = np.empty((grid_height, grid_width))
        self.current_grid = np.empty((grid_height, grid_width))
        
    @staticmethod
    def from_strings(string_array, dead_cell = "_", live_cell = "X"):
        grid_height = len(string_array)
        grid_width = len(string_array[0])
        
        serialized_integer_array = ",".join("".join(string_array)).replace(dead_cell, "0").replace(live_cell, "1")
        new_grid = np.fromstring(serialized_integer_array, dtype=int, sep=",").reshape(grid_height, grid_width)
        
        new_environment = ConwaysEnvironment(grid_height, grid_width)    
        new_environment.current_grid = new_grid
        
        return new_environment
        
    def randomize(self, live_probability = 0.2):
        self.current_grid = np.random.choice(
            [1, 0],
            self.grid_height * self.grid_width,
            p=[live_probability, 1 - live_probability]
        ).reshape(self.grid_height, self.grid_width)

    def iterate(self):
        # Triangle swap the grid references so that current grid is now the previous grid.
        tmp_grid = self.previous_grid
        self.previous_grid = self.current_grid
        self.current_grid = tmp_grid

        grid_height = self.grid_height
        grid_width = self.grid_width

        # Overwrite the current grid based on the previous grid contents.
        for row in range(grid_height):
            for col in range(grid_width):
                live_neighbours = (
                    self.previous_grid[row, (col - 1) % grid_width] +
                    self.previous_grid[row, (col + 1) % grid_width] +
                    self.previous_grid[(row - 1) % grid_height, col] +
                    self.previous_grid[(row + 1) % grid_height, col] +
                    self.previous_grid[(row - 1) % grid_height, (col - 1) % grid_width] +
                    self.previous_grid[(row - 1) % grid_height, (col + 1) % grid_width] +
                    self.previous_grid[(row + 1) % grid_height, (col - 1) % grid_width] +
                    self.previous_grid[(row + 1) % grid_height, (col + 1) % grid_width]
                )
                if self.previous_grid[row, col] == 1:
                    if live_neighbours == 2 or live_neighbours == 3:
                        self.current_grid[row, col] = 1
                    else:
                        self.current_grid[row, col] = 0
                else:
                    if live_neighbours == 3:
                        self.current_grid[row, col] = 1
                    else:
                        self.current_grid[row, col] = 0