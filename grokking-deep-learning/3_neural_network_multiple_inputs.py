from typing import List


def neural_network(input_values: List[float], weights: List[float]) -> float:
    return w_sum(input_values, weights)


def w_sum(a: List[float], b: List[float]) -> float:
    assert (len(a) == len(b))
    result = 0
    for i in range(len(a)):
        result += (a[i] * b[i])
    return result


def main():
    game_counts = [8.5, 9.5, 9.9, 9.0]
    win_shares = [0.65, 0.8, 0.8, 0.9]
    fan_counts = [1.2, 1.3, 0.5, 1.0]

    weights = [0.1, 0.2, 0]
    input_values = [game_counts[0], win_shares[0], fan_counts[0]]
    prediction = neural_network(input_values, weights)
    print(prediction)


if __name__ == '__main__':
    main()
