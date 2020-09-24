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
        self.max_fitness = 0
        # If no weights provided randomly generate them
        if child_weights is None:
            # Layers are created and randomly generated
            layer1 = Dense(self.i_size, input_shape=(self.i_size,), activation='tanh', bias_initializer='random_normal')
            layer2 = Dense(self.h1_size, activation='tanh')
            layer3 = Dense(self.h2_size, activation='tanh')
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
                    activation='tanh',
                    weights=[child_weights[0], np.zeros(self.i_size)])
            )
            self.add(
                Dense(
                    self.h1_size,
                    activation='tanh',
                    weights=[child_weights[1], np.zeros(self.h1_size)])
            )
            self.add(
                Dense(
                    self.h2_size,
                    activation='tanh',
                    weights=[child_weights[2], np.zeros(self.h2_size)])
            )
            self.add(
                Dense(
                    self.o_size,
                    activation='softmax',
                    weights=[child_weights[3], np.zeros(self.o_size)])
            )

    def fitness_update(self, distance):
        self.fitness = (1 - distance / 28.2842712475)


# Chance to mutate weights
def mutation(child_weights):
    # Add a chance for random mutation
    selection = random.randint(0, len(child_weights) - 1)
    mut = random.uniform(0, 1)
    if mut >= .5:
        child_weights[selection] *= random.random()*2
    else:
        # No mutation
        pass


# Crossover traits between two Genetic Neural Networks
def dynamic_crossover(nn1, nn2):
    # A new child is born
    child = GeneticANN()
    child.compile()
    child_weights_l1 = [np.zeros((5, 5)), np.zeros(5, )]
    child_weights_l2 = [np.zeros((5, 5)), np.zeros(5, )]
    child_weights_l3 = [np.zeros((5, 5)), np.zeros(5, )]
    child_weights_l4 = [np.zeros((5, 4)), np.zeros(4, )]

    # Layer 1
    for w_I in range(0, nn1.i_size):
        for w_J in range(0, nn1.i_size):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 15:
                child_weights_l1[0][w_I, w_J] = nn1.layers[0].kernel[w_I, w_J]
            else:
                child_weights_l1[0][w_I, w_J] = nn2.layers[0].kernel[w_I, w_J]
    for b_I in range(0, nn1.i_size):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 15:
            child_weights_l1[1][b_I] = nn1.layers[0].bias[b_I]
        else:
            child_weights_l1[1][b_I] = nn2.layers[0].bias[b_I]
    mutation(child_weights_l1)
    child.layers[0].set_weights(child_weights_l1)

    # Layer 2
    for w_I in range(0, nn1.i_size):
        for w_J in range(0, nn1.h1_size):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 15:
                child_weights_l2[0][w_I, w_J] = nn1.layers[1].kernel[w_I, w_J]
            else:
                child_weights_l2[0][w_I, w_J] = nn2.layers[1].kernel[w_I, w_J]
    for b_I in range(0, nn1.h1_size):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 15:
            child_weights_l2[1][b_I] = nn1.layers[1].bias[b_I]
        else:
            child_weights_l2[1][b_I] = nn2.layers[1].bias[b_I]
    mutation(child_weights_l2)
    child.layers[1].set_weights(child_weights_l2)

    # Layer 3
    for w_I in range(0, nn1.h1_size):
        for w_J in range(0, nn1.h2_size):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 15:
                child_weights_l3[0][w_I, w_J] = nn1.layers[2].kernel[w_I, w_J]
            else:
                child_weights_l3[0][w_I, w_J] = nn2.layers[2].kernel[w_I, w_J]
    for b_I in range(0, nn1.h2_size):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 15:
            child_weights_l3[1][b_I] = nn1.layers[2].bias[b_I]
        else:
            child_weights_l3[1][b_I] = nn2.layers[2].bias[b_I]
    mutation(child_weights_l3)
    child.layers[2].set_weights(child_weights_l3)

    # Layer 4
    for w_I in range(0, nn1.h2_size):
        for w_J in range(0, nn1.o_size):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 15:
                child_weights_l4[0][w_I, w_J] = nn1.layers[3].kernel[w_I, w_J]
            else:
                child_weights_l4[0][w_I, w_J] = nn2.layers[3].kernel[w_I, w_J]
    for b_I in range(0, nn1.o_size):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 15:
            child_weights_l4[1][b_I] = nn1.layers[3].bias[b_I]
        else:
            child_weights_l4[1][b_I] = nn2.layers[3].bias[b_I]
    mutation(child_weights_l4)
    child.layers[3].set_weights(child_weights_l4)

    return child
