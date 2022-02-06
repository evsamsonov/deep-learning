import numpy as np


def main():
    alpha = 0.1
    weights = np.array([0.5, 0.48, -0.7])

    street_lights = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 0, 1],
    ])

    walk_or_stop = np.array([
        [0],
        [1],
        [0],
        [1],
        [1],
        [0]
    ])

    for i in range(40):
        for j in range(len(walk_or_stop)):
            input_values = street_lights[j]
            goal_prediction = walk_or_stop[j]

            prediction = input_values.dot(weights)
            error = (goal_prediction - prediction) ** 2
            delta = prediction - goal_prediction

            weights = weights - (alpha * (input_values * delta))

            print("Iteration: " + str(i) + ", Row:" + str(j))
            print("Prediction: " + str(prediction) + " Error: " + str(error))
            print("Weights: " + str(weights))
        print("\n")


if __name__ == '__main__':
    main()
