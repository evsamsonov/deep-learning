import numpy as np


def neural_network(input_value: float, weights: np.ndarray) -> np.ndarray:
    # Умножаем каждый элемент вектора на число и получаем вектор с результатами
    return weights * input_value


def main():
    # Веса
    weights = np.array([0.3, 0.2, 0.9])

    # Доля побед
    win_shares = np.array([0.65, 0.8, 0.8, 0.9])

    # Реальные значения
    hurt = [0.1, 0.0, 0.0, 0.1]  # Травм
    win = [1, 0.0, 0.0, 0.1]   # Побед
    sad = [0.1, 0.0, 0.1, 0.2]   # Печаль
    real_values = np.array([hurt[0], win[0], sad[0]])

    input_value = win_shares[0]

    prediction = neural_network(input_value, weights)
    print("Prediction: " + str(prediction))

    errors = [0, 0, 0]
    deltas = [0, 0, 0]

    for i in range(len(real_values)):
        errors[i] = (prediction[i] - real_values[i]) ** 2
        deltas[i] = prediction[i] - real_values[i]

        print("Errors: " + str(errors))
        print("Deltas: " + str(deltas))

    weight_deltas = input_value * np.array(deltas)
    alpha = 0.1

    for i in range(len(weights)):
        weights[i] -= (weight_deltas[i] * alpha)

    print("Weights: " + str(weights))
    print("Weight Deltas: " + str(weight_deltas))


if __name__ == '__main__':
    main()
