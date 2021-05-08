import numpy as np


def neural_network(input_value: float, weights: np.ndarray) -> np.ndarray:
    # Умножаем каждое элемент вектора на число и получаем вектор с результатом
    return weights * input_value


def main():
    win_shares = np.array([0.65, 0.8, 0.8, 0.9])

    weights = np.array([0.3, 0.2, 0.9])
    input_value = win_shares[0]
    prediction = neural_network(input_value, weights)
    print(prediction)


if __name__ == '__main__':
    main()
