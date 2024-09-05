import math
import random

"""
Problem to Solve
              Input      Output
Example 1:  0   0   1       0
Example 2:  1   1   1       1
Example 3:  1   0   1       1
Example 4:  0   1   1       0

New Input:  1   0   0       ?
"""

class NeuralNetwork():
    def __init__(self):
        random.seed(1) # Seed random num generator
        self.weights = [random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)]

    # Make a prediction
    def think(self, neuron_inputs):
        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        neuron_output = self.__sigmoid_function(sum_of_weighted_inputs)
        return neuron_output

    # Where the weights are adjusted to minimize the error
    def train(self, training_set_data, number_of_iterations):
        for _ in range(number_of_iterations):
            for training_set in training_set_data:
                # Predict output
                predicted_output = self.think(training_set["inputs"])
                # Calc the error between desired output and predicted output
                error_in_output = training_set["output"] - predicted_output

                # Adjust the weights to min total error
                for index in range(len(self.weights)):
                    # Get neuron input
                    neuron_input = training_set["inputs"][index]
                    # Calc weight adjustment via delta rule (gradient descent): Input * ErrorInOutput * SigmoidCurveGradient
                    adjusted_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)
                    # Update weight with adjusted weights
                    self.weights[index] += adjusted_weight

    # Calc the sigmoid activation function 
    def __sigmoid_function(self, sum_of_weighted_inputs):
        # 1 / (1 + e^-x)
        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

    # Calc the gradient of the sigmoid function 
    def __sigmoid_gradient(self, neuron_output):
        # Sigmoid Gradient = NeuronOutput * (1 - NeuronOutput)
        return neuron_output * (1 - neuron_output)

    # Multiple each input by its weight
    def __sum_of_weighted_inputs(self, neuron_inputs):
        # Σ(Input_i * Weight_i)
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs

# Data used to train the NN
training_set_data = [
                    {"inputs": [0, 0, 1], "output": 0}, 
                    {"inputs": [1, 1, 1], "output": 1},
                    {"inputs": [1, 0, 1], "output": 1}, 
                    {"inputs": [0, 1, 1], "output": 0}
                    ]

neural_network = NeuralNetwork()
print("Starting weights: ", str(neural_network.weights), end='\n\n')

# Train the NN with 10,000 iterations
number_of_iterations = 10000
neural_network.train(training_set_data, number_of_iterations)
print("Trained weights: ", str(neural_network.weights), end='\n\n')

new_input = [1, 0, 0]
prediction = neural_network.think(new_input)
print("prediction for [1, 0, 0]: ", str(prediction), end='\n')