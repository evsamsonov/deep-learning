def neural_network( input_value: float, weight: float) -> float:
    return weight * input_value


def get_error(prediction: float, real: float) -> float:
    return (prediction - real) ** 2


def main():
    weight = 0.5
    input_value = 0.5
    real = 0.8

    step = 0.001
    for i in range(1101):
        prediction = neural_network(input_value, weight)
        mead_squared_error = get_error(prediction, real)
        print(
            "Error: " + str(mead_squared_error) +
            " Prediction: " + str(prediction) +
            " Weight: " + str(weight)
        )

        up_prediction = neural_network(input_value, weight + step)
        up_error = get_error(up_prediction, real)

        down_prediction = neural_network(input_value, weight - step)
        down_error = get_error(down_prediction, real)

        if down_error < up_error:
            weight -= step
        else:
            weight += step


if __name__ == '__main__':
    main()
