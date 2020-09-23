#
# Created by Ronny Milleo based on:
# github.com/RomanMichaelPaolucci/Genetic_Neural_Network
#

import random

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# New Type of Neural Network
class GeneticANN(Sequential):
    # Inputs
    # Player x, y
    # Target x, y
    # Distance
    i_size = 5
    h1_size = 5
    h2_size = 5
    o_size = 4

    # Outputs
    # Move Up, Down, Left, Right

    # Constructor
    def __init__(self, child_weights=None):
        # Initialize Sequential Model Super Class
        super().__init__()
        self.fitness = 0
        # If no weights provided randomly generate them
        if child_weights is None:
            # Layers are created and randomly generated
            layer1 = Dense(self.i_size, input_shape=(self.i_size,), activation='sigmoid')
            layer2 = Dense(self.h1_size, activation='sigmoid')
            layer3 = Dense(self.h2_size, activation='sigmoid')
            layer4 = Dense(self.o_size, activation='softmax')
            # Layers are added to the model
            self.add(layer1)
            self.add(layer2)
            self.add(layer3)
            self.add(layer4)
        # If weights are provided set them within the layers
        else:
            # Set weights within the layers
            self.add(
                Dense(
                    self.i_size,
                    input_shape=(self.i_size,),
                    activation='sigmoid',
                    weights=[child_weights[0], np.zeros(self.i_size)])
            )
            self.add(
                Dense(
                    self.h1_size,
                    activation='sigmoid',
                    weights=[child_weights[1], np.zeros(self.h1_size)])
            )
            self.add(
                Dense(
                    self.h2_size,
                    activation='sigmoid',
                    weights=[child_weights[2], np.zeros(self.h2_size)])
            )
            self.add(
                Dense(
                    self.o_size,
                    activation='softmax',
                    weights=[child_weights[3], np.zeros(self.o_size)])
            )

    # # Function for forward propagating a row vector of a matrix
    # def forward_propagation(self, X_train, y_train):
    #     # Forward propagation
    #     y_h = self.predict(X_train.values)
    #     # Compute fitness score
    #     self.fitness = accuracy_score(y_train, y_h.round())

    def fitness_update(self, distance):
        self.fitness = (1 - distance / 10)


# # Standard Backpropagation
# def compile_train(nn, X_train, y_train, epochs):
#     nn.compile(
#         optimizer='rmsprop',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
#     nn.fit(X_train.values, y_train.values, epochs=epochs)


# Chance to mutate weights
def mutation(child_weights):
    # Add a chance for random mutation
    selection = random.randint(0, len(child_weights) - 1)
    mut = random.uniform(0, 1)
    if mut >= .3:
        child_weights[selection] *= random.random()
    else:
        # No mutation
        pass

    # Crossover traits between two Genetic Neural Networks


def dynamic_crossover(nn1, nn2):
    # Lists for respective weights
    nn1_weights = []
    nn2_weights = []
    child_weights = []
    # Get all weights from all layers in the first network
    for lyr in nn1.layers:
        nn1_weights.append(lyr.get_weights()[0])

    # Get all weights from all layers in the second network
    for lyr in nn2.layers:
        nn2_weights.append(lyr.get_weights()[0])

    # Iterate through all weights from all layers for crossover
    for a in range(0, len(nn1_weights)):
        # Get single point to split the matrix in parents based on # of cols
        split = random.randint(0, np.shape(nn1_weights[a])[1] - 1)
        # Iterate through after a single point and set the remaining cols to nn_2
        for b in range(split, np.shape(nn1_weights[a])[1] - 1):
            nn1_weights[a][:, b] = nn2_weights[a][:, b]

        # After crossover add weights to child
        child_weights.append(nn1_weights[a])

    # Add a chance for mutation
    mutation(child_weights)

    # Create and return child object
    child = GeneticANN(child_weights)
    return child
