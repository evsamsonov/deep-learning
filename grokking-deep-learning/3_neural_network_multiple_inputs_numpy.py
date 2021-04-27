import numpy as np


def neural_network(input_values: np.ndarray, weights: np.ndarray) -> float:
    return input_values.dot(weights)


def main():
    game_counts = np.array([8.5, 9.5, 9.9, 9.0])
    win_shares = np.array([0.65, 0.8, 0.8, 0.9])
    fan_counts = np.array([1.2, 1.3, 0.5, 1.0])

    weights = np.array([0.1, 0.2, 0])
    input_values = np.array([game_counts[0], win_shares[0], fan_counts[0]])
    prediction = neural_network(input_values, weights)
    print(prediction)


if __name__ == '__main__':
    main()
