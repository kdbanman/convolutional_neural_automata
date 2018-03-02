import numpy as np

class ConwaysEnvironment:
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        
        self.grid = np.empty((grid_size, grid_size))
        self.next_grid = np.empty((grid_size, grid_size))
        
    def randomize(self, live_probability = 0.2):
        self.grid = np.random.choice(
            [1, 0],
            self.grid_size * self.grid_size,
            p=[live_probability, 1 - live_probability]
        ).reshape(self.grid_size, self.grid_size)

    def iterate(self):
        grid_size = self.grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                live_neighbours = (
                    self.grid[i, (j - 1) % grid_size] +
                    self.grid[i, (j + 1) % grid_size] +
                    self.grid[(i - 1) % grid_size, j] +
                    self.grid[(i + 1) % grid_size, j] +
                    self.grid[(i - 1) % grid_size, (j - 1) % grid_size] +
                    self.grid[(i - 1) % grid_size, (j + 1) % grid_size] +
                    self.grid[(i + 1) % grid_size, (j - 1) % grid_size] +
                    self.grid[(i + 1) % grid_size, (j + 1) % grid_size]
                )
                if self.grid[i, j] == 1:
                    if live_neighbours == 2 or live_neighbours == 3:
                        self.next_grid[i, j] = 1
                    else:
                        self.next_grid[i, j] = 0
                else:
                    if live_neighbours == 3:
                        self.next_grid[i, j] = 1
                    else:
                        self.next_grid[i, j] = 0

        tmp_grid = self.grid
        self.grid = self.next_grid
        self.next_grid = tmp_grid