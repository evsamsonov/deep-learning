import numpy as np


def neural_network(input_values, weights):
    assert(len(input_values) == len(weights))
    output = [0, 0, 0]

    # Получаем скалярное произведение вектора с входным значением
    # на каждый вектор весов
    for i in range(len(weights)):
        output[i] = input_values.dot(weights[i])
    return output


def main():
    # Веса
    weights = np.array([
        # Кол-во игр, % победа/поражение, кол-во болельщиков
        [0.1, 0.1, -0.3],   # Травмы
        [0.1, 0.2, 0.0],    # Победы
        [0.0, 1.3, 0.1],    # Печаль
    ])

    game_counts = [8.5, 9.5, 9.9, 9.0]
    win_shares = [0.65, 0.8, 0.8, 0.9]
    fan_counts = [1.2, 1.3, 0.5, 1.0]

    # Реальные значения
    hurt = [0.1, 0.0, 0.0, 0.1]  # Травм
    win = [1, 0.0, 0.0, 0.1]   # Побед
    sad = [0.1, 0.0, 0.1, 0.2]   # Печаль

    input_values = np.array([game_counts[0], win_shares[0], fan_counts[0]])
    real_values = np.array([hurt[0], win[0], sad[0]])

    prediction = neural_network(input_values, weights)
    print("Prediction: " + str(prediction))

    errors = [0, 0, 0]
    deltas = [0, 0, 0]
    for i in range(len(real_values)):
        errors[i] = (prediction[i] - real_values[i]) ** 2
        deltas[i] = prediction[i] - real_values[i]

    print("Errors: " + str(errors))
    print("Deltas: " + str(deltas))

    # Перемножаем каждое входное занчение на каждое delta
    weight_deltas = np.zeros((len(deltas), len(input_values)))
    for i in range(len(deltas)):
        for j in range(len(input_values)):
            weight_deltas[i][j] = deltas[i] * input_values[j]
    print("Weight Deltas: " + str(weight_deltas))

    # Корректируем веса
    alpha = 0.01
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            weights[i][j] -= (alpha * weight_deltas[i][j])

    print("Weights: " + str(weights))


if __name__ == '__main__':
    main()
