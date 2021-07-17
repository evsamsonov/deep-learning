import numpy as np


def neural_network(input_values: np.ndarray, weights: np.ndarray) -> float:
    return input_values.dot(weights)


def get_error(prediction: float, real: float) -> float:
    return (prediction - real) ** 2


def main():
    game_counts = [8.5, 9.5, 9.9, 9.0]
    win_shares = [0.65, 0.8, 0.8, 0.9]
    fan_counts = [1.2, 1.3, 0.5, 1.0]
    weights = np.array([0.1, 0.2, -0.1])

    win_binary = [1, 1, 0, 1]

    input_values = np.array([game_counts[0], win_shares[0], fan_counts[0]])

    prediction = neural_network(input_values, weights)

    error = get_error(prediction, win_binary[0])
    delta = prediction - win_binary[0]
    weights_delta = input_values * delta

    print(
        "Prediction: " + str(prediction) + "\n"
        "Error: " + str(error) + "\n"
        "Delta: " + str(delta) + "\n"
        "Weights delta: " + str(weights_delta)
    )

    alpha = 0.01
    for i in range(len(weights)):
        weights[i] -= alpha * weights_delta[i]

    print(
        "After correction\n"
        "Weights: " + str(weights)
    )


if __name__ == '__main__':
    main()
