import numpy as np

class ConwaysEnvironment:
    
    def __init__(self, grid_height, grid_width):
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        self.grid = np.empty((grid_height, grid_width))
        self.next_grid = np.empty((grid_height, grid_width))
        
    @staticmethod
    def from_strings(string_array, dead_cell = "_", live_cell = "X"):
        grid_height = len(string_array)
        grid_width = len(string_array[0])
        
        serialized_integer_array = ",".join("".join(string_array)).replace(dead_cell, "0").replace(live_cell, "1")
        new_grid = np.fromstring(serialized_integer_array, dtype=int, sep=",").reshape(grid_height, grid_width)
        
        new_environment = ConwaysEnvironment(grid_height, grid_width)    
        new_environment.grid = new_grid
        
        return new_environment
        
    def randomize(self, live_probability = 0.2):
        self.grid = np.random.choice(
            [1, 0],
            self.grid_height * self.grid_width,
            p=[live_probability, 1 - live_probability]
        ).reshape(self.grid_height, self.grid_width)

    def iterate(self):
        grid_height = self.grid_height
        grid_width = self.grid_width

        for row in range(grid_height):
            for col in range(grid_width):
                live_neighbours = (
                    self.grid[row, (col - 1) % grid_width] +
                    self.grid[row, (col + 1) % grid_width] +
                    self.grid[(row - 1) % grid_height, col] +
                    self.grid[(row + 1) % grid_height, col] +
                    self.grid[(row - 1) % grid_height, (col - 1) % grid_width] +
                    self.grid[(row - 1) % grid_height, (col + 1) % grid_width] +
                    self.grid[(row + 1) % grid_height, (col - 1) % grid_width] +
                    self.grid[(row + 1) % grid_height, (col + 1) % grid_width]
                )
                if self.grid[row, col] == 1:
                    if live_neighbours == 2 or live_neighbours == 3:
                        self.next_grid[row, col] = 1
                    else:
                        self.next_grid[row, col] = 0
                else:
                    if live_neighbours == 3:
                        self.next_grid[row, col] = 1
                    else:
                        self.next_grid[row, col] = 0

        tmp_grid = self.grid
        self.grid = self.next_grid
        self.next_grid = tmp_grid