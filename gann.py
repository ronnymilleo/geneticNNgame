#
# Created by Ronny Milleo based on:
# github.com/RomanMichaelPaolucci/Genetic_Neural_Network
#

import random

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense

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
    # Quadrant
    # Angle
    i_size = 8
    h1_size = 8
    h2_size = 6
    o_size = 4

    # Outputs
    # Move Up, Down, Left, Right

    # Constructor
    def __init__(self):
        # Initialize Sequential Model Super Class
        super().__init__()
        self.fitness = 0
        self.fitness_array = []
        self.max_fitness = 0
        self.mean_fitness = 0
        # If no weights provided randomly generate them
        # Layers are created and randomly generated
        layer1 = Dense(self.i_size, input_shape=(self.i_size,), activation='tanh', bias_initializer='random_normal')
        layer2 = Dense(self.h1_size, activation='tanh', bias_initializer='random_normal')
        layer3 = Dense(self.h2_size, activation='tanh', bias_initializer='random_normal')
        layer4 = Dense(self.o_size, activation='softmax')
        # Layers are added to the model
        self.add(layer1)
        self.add(layer2)
        self.add(layer3)
        self.add(layer4)

    def fitness_update(self, distance):
        self.fitness = (1 - distance / 26.87)
        self.fitness_array.append(self.fitness)
        self.mean_fitness = np.mean(self.fitness_array)


# Chance to mutate weights
def mutation(child_weights):
    # Add a chance for random mutation
    selection_1 = random.randint(0, len(child_weights) - 1)
    selection_2 = random.randint(0, len(child_weights) - 1)
    mut = random.uniform(0, 1)
    if mut >= .4:
        child_weights[selection_1] *= random.random()*1.5
        child_weights[selection_2] *= random.random()*0.5
    else:
        # No mutation
        pass


# Crossover traits between two Genetic Neural Networks
def dynamic_crossover(parent_1, parent_2) -> GeneticANN:
    # A new child is born
    child = GeneticANN()
    child_weights_l1 = [np.zeros((parent_1.i_size, parent_1.i_size)), np.zeros(parent_1.i_size, )]
    child_weights_l2 = [np.zeros((parent_1.i_size, parent_1.h1_size)), np.zeros(parent_1.h1_size, )]
    child_weights_l3 = [np.zeros((parent_1.h1_size, parent_1.h2_size)), np.zeros(parent_1.h2_size, )]
    child_weights_l4 = [np.zeros((parent_1.h2_size, parent_1.o_size)), np.zeros(parent_1.o_size, )]

    # Layer 1
    for w_I in range(0, parent_1.i_size):
        for w_J in range(0, parent_1.i_size):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 10:
                child_weights_l1[0][w_I, w_J] = parent_1.layers[0].kernel[w_I, w_J]
            else:
                child_weights_l1[0][w_I, w_J] = parent_2.layers[0].kernel[w_I, w_J]
    for b_I in range(0, parent_1.i_size):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 10:
            child_weights_l1[1][b_I] = parent_1.layers[0].bias[b_I]
        else:
            child_weights_l1[1][b_I] = parent_2.layers[0].bias[b_I]
    mutation(child_weights_l1)
    child.layers[0].set_weights(child_weights_l1)

    # Layer 2
    for w_I in range(0, parent_1.i_size):
        for w_J in range(0, parent_1.h1_size):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 10:
                child_weights_l2[0][w_I, w_J] = parent_1.layers[1].kernel[w_I, w_J]
            else:
                child_weights_l2[0][w_I, w_J] = parent_2.layers[1].kernel[w_I, w_J]
    for b_I in range(0, parent_1.h1_size):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 10:
            child_weights_l2[1][b_I] = parent_1.layers[1].bias[b_I]
        else:
            child_weights_l2[1][b_I] = parent_2.layers[1].bias[b_I]
    mutation(child_weights_l2)
    child.layers[1].set_weights(child_weights_l2)

    # Layer 3
    for w_I in range(0, parent_1.h1_size):
        for w_J in range(0, parent_1.h2_size):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 10:
                child_weights_l3[0][w_I, w_J] = parent_1.layers[2].kernel[w_I, w_J]
            else:
                child_weights_l3[0][w_I, w_J] = parent_2.layers[2].kernel[w_I, w_J]
    for b_I in range(0, parent_1.h2_size):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 10:
            child_weights_l3[1][b_I] = parent_1.layers[2].bias[b_I]
        else:
            child_weights_l3[1][b_I] = parent_2.layers[2].bias[b_I]
    mutation(child_weights_l3)
    child.layers[2].set_weights(child_weights_l3)

    # Layer 4
    for w_I in range(0, parent_1.h2_size):
        for w_J in range(0, parent_1.o_size):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 10:
                child_weights_l4[0][w_I, w_J] = parent_1.layers[3].kernel[w_I, w_J]
            else:
                child_weights_l4[0][w_I, w_J] = parent_2.layers[3].kernel[w_I, w_J]
    for b_I in range(0, parent_1.o_size):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 10:
            child_weights_l4[1][b_I] = parent_1.layers[3].bias[b_I]
        else:
            child_weights_l4[1][b_I] = parent_2.layers[3].bias[b_I]
    mutation(child_weights_l4)
    child.layers[3].set_weights(child_weights_l4)

    return child


def dna_exam(parent_1: GeneticANN, parent_2: GeneticANN, child: GeneticANN):
    p1_layer_1 = parent_1.layers[0].get_weights()[0]
    p1_layer_2 = parent_1.layers[1].get_weights()
    p1_layer_3 = parent_1.layers[2].get_weights()
    p1_layer_4 = parent_1.layers[3].get_weights()
    p2_layer_2 = parent_2.layers[0].get_weights()
    p2_layer_2 = parent_2.layers[1].get_weights()
    p2_layer_3 = parent_2.layers[2].get_weights()
    p2_layer_4 = parent_2.layers[3].get_weights()
    ch_layer_1 = child.layers[0].get_weights()[0]
    ch_layer_2 = child.layers[1].get_weights()
    ch_layer_3 = child.layers[2].get_weights()
    ch_layer_4 = child.layers[3].get_weights()
    return np.array(ch_layer_1) - np.array(p1_layer_1)
