import time

import matplotlib.pyplot as plt

GRID_SHAPE = (512, 512)


@profile
def evolve(grid, dt, new_grid, D=1.0):

    xmax, ymax = GRID_SHAPE

    for i in range(xmax):
        for j in range(ymax):
            grid_xx = grid[(i+1)%xmax][j] + grid[(i-1)%xmax][j] - 2.0 * grid[i][j]
            grid_yy = grid[i][(j+1)%ymax] + grid[i][(j-1)%ymax] - 2.0 * grid[i][j]

            new_grid[i][j] = grid[i][j] + D * (grid_xx + grid_yy) * dt

    return new_grid


def run_experiment(num_iterations):

    xmax, ymax = GRID_SHAPE
    grid = [[0.0,] * ymax for x in range(xmax)]
    new_grid = [[0.0,] * ymax for x in range(xmax)]

    block_low = int(GRID_SHAPE[0] * 0.4)
    block_high = int(GRID_SHAPE[0] * 0.5)

    for i in range(block_low, block_high):
        for j in range(block_low, block_high):
            grid[i][j] = 0.005

    # Evolve the initial conditions
    start = time.time()
    for i in range(num_iterations):
        grid = evolve(grid, 0.1, new_grid)

    return time.time() - start


if __name__ == "__main__":
    run_experiment(10)
