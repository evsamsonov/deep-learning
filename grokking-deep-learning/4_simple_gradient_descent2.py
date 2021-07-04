def neural_network( input_value: float, weight: float) -> float:
    return weight * input_value


def get_error(prediction: float, real: float) -> float:
    return (prediction - real) ** 2


def main():
    weight, goal_prediction, input_value = (0., 0.8, 1.1)

    for i in range(4):
        print("Weight: " + str(weight))

        prediction = neural_network(input_value, weight)
        error = (prediction - goal_prediction) ** 2

        delta = prediction - goal_prediction
        weight_delta = delta * input_value
        weight -= weight_delta

        print(
            "Error: " + str(error) +
            " Prediction: " + str(prediction) +
            " Delta: " + str(delta) +
            " Weight delta: " + str(weight_delta)
        )


if __name__ == '__main__':
    main()
