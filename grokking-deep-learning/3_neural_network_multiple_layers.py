import numpy as np

def multiple(input_values: np.ndarray, weights: np.ndarray):
    # Умножаем входящий вектор на каждый вектор в матрице весов
    output = [0, 0, 0]
    for i in range(len(input_values)):
        output[i] = input_values.dot(weights[i])
    return np.array(output)


def neural_network(input_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    hidden = multiple(input_values, weights[0])
    prediction = multiple(hidden, weights[1])
    return prediction


def main():
    game_counts = [8.5, 9.5, 9.9, 9.0]
    win_shares = [0.65, 0.8, 0.8, 0.9]
    fan_counts = [1.2, 1.3, 0.5, 1.0]

    weights1 = np.array([
        [0.1, 0.2, -0.1],  # Травмы?
        [-0.1, 0.1, 0.9],  # Победы?
        [0.1, 0.4, 0.1]  # Печаль?
    ])
    weights2 = np.array([
        [0.3, 1.1, -0.3],  # Травмы?
        [0.1, 0.2, 0.0],  # Победы?
        [0.0, 1.3, 0.1]  # Печаль?
    ])

    weights = np.array([weights1, weights2])
    input_values = np.array([game_counts[0], win_shares[0], fan_counts[0]])
    prediction = neural_network(input_values, weights)
    print(prediction)


if __name__ == '__main__':
    main()
