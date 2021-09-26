import numpy as np


def main():
    weights = np.array([0.5, 0.48, -0.7])

    street_lights = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
    ])

    walk_or_stop = np.array([0, 1, 0, 1, 1, 0])


if __name__ == '__main__':
    main()
