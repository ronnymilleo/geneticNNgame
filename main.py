import math
import os.path
import random

import matplotlib
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pygame

import gann
from player import Player


# Drawing functions
def draw_target():
    screen.blit(target_icon, (t_px_pos_x, t_px_pos_y))


def draw_flag():
    screen.blit(flag_icon, (f_px_pos_x, f_px_pos_y))


def draw_grid():
    for x in range(left_margin, left_margin + right_margin + block_size, block_size):
        pygame.draw.line(screen, (0, 0, 0), (x, up_margin), (x, height - down_margin), 1)
    for y in range(up_margin, height - down_margin + block_size, block_size):
        pygame.draw.line(screen, (0, 0, 0), (left_margin, y), (left_margin + right_margin, y), 1)


def draw_text():
    gen = font.render("Generation = {}".format(generation), True, [0, 0, 0], [255, 255, 255])
    pop = font.render("Population = {}".format(population), True, [0, 0, 0], [255, 255, 255])
    steps = font.render("Steps = {}".format(step), True, [0, 0, 0], [255, 255, 255])
    dist = font.render("Distance = {:.2f}".format(distance), True, [0, 0, 0], [255, 255, 255])
    position = font.render("Position = ({}, {})".format(players[player].pos_x, players[player].pos_y), True, [0, 0, 0],
                           [255, 255, 255])
    screen.blit(gen, (20, 20))
    screen.blit(pop, (20 + 150, 20))
    screen.blit(steps, (20 + 300, 20))
    screen.blit(dist, (20 + 400, 20))
    screen.blit(position, (20 + 550, 20))


def draw_fit_table():
    player_text = font.render("Player", True, [0, 0, 0], [255, 255, 255])
    fitness_text = font.render("Fitness", True, [0, 0, 0], [255, 255, 255])
    screen.blit(player_text, (720, 60))
    screen.blit(fitness_text, (720 + 100, 60))
    for p in range(population):
        player_id = font.render("P{}".format(p), True, [0, 0, 0], [255, 255, 255])
        fitness_number = font.render("{:.4f}".format(players[p].neural_network.max_fitness), True, [0, 0, 0],
                                     [255, 255, 255])
        screen.blit(player_id, (720, 60 + p * 24 + 24))
        screen.blit(fitness_number, (720 + 100, 60 + p * 24 + 24))


def plot(data):
    ax.plot(data)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    return pygame.image.fromstring(raw_data, size, "RGB")


def draw_player(pos_x, pos_y):
    player_rect = pygame.Surface((block_size, block_size))
    player_rect.fill((127, 0, 127))
    player_rect.set_alpha(128)
    screen.blit(player_rect, (pos_x, pos_y))


def render():
    screen.fill((255, 255, 255))
    draw_text()
    draw_grid()
    draw_flag()
    draw_target()
    draw_fit_table()
    draw_player(p_px_pos_x, p_px_pos_y)
    screen.blit(surf, (720 + 200, 60))
    pygame.display.update()
    clock.tick(60)


# Calculations and converting functions
def euclidian_distance(dest_pos_x, dest_pos_y, origin_pos_x, origin_pos_y):
    new_distance = math.sqrt(math.pow(dest_pos_x - origin_pos_x, 2) + math.pow(dest_pos_y - origin_pos_y, 2))
    return new_distance


def x_conv(px_x):
    return (px_x - left_margin) // 32 + 1


def y_conv(px_y):
    return 20 - (px_y - up_margin) // 32


def random_position():
    return math.floor(random.random() * (640 / block_size)) * block_size + left_margin, \
           math.floor(random.random() * (640 / block_size)) * block_size + up_margin


if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()

    # Plot learning graph
    matplotlib.use("Agg")
    fig = plt.figure(figsize=[6, 5], dpi=80)
    ax = fig.add_subplot(111)
    canvas = agg.FigureCanvasAgg(fig)
    history = np.array([])
    surf = plot(history)

    # Screen globals
    width = 1440
    height = 720
    block_size = 32
    left_margin = 40
    right_margin = block_size * 20
    up_margin = 60
    down_margin = 20

    # Game display
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mistery's game")
    icon = pygame.image.load('deep-learning.png')
    pygame.display.set_icon(icon)
    flag_icon = pygame.image.load('start_flag.png')
    target_icon = pygame.image.load('target.png')
    font = pygame.font.Font(os.path.curdir + '\\font\\Roboto-Regular.ttf', 20)

    # Max generations and initial population
    generations = 100
    population = 20
    # Generate n randomly weighted neural networks
    # Create a List of all active GeneticNeuralNetworks
    players = []
    # Save movements for later
    players_movement_list = []
    # Aux pool to make crossover and mutation
    pool = []
    for i in range(0, population):
        players.append(Player(0, 0))
    # Cache Max Fitness
    max_fitness = 0

    # Each generation has the same start and finish points
    # Generate start position
    f_px_pos_x, f_px_pos_y = random_position()
    p_px_pos_x, p_px_pos_y = f_px_pos_x, f_px_pos_y

    # Generate target's position and check if target is at same position as start flag
    t_px_pos_x, t_px_pos_y = random_position()
    while abs(t_px_pos_x - p_px_pos_x) < block_size * 5 and abs(t_px_pos_y - p_px_pos_y) < block_size * 5:
        t_px_pos_x, t_px_pos_y = random_position()

    # Main loop
    for generation in range(0, generations):
        # Every 10 generations, change the target's position
        div, res = divmod(generation, 10)
        if res == 0:
            f_px_pos_x, f_px_pos_y = random_position()
            p_px_pos_x, p_px_pos_y = f_px_pos_x, f_px_pos_y
            t_px_pos_x, t_px_pos_y = random_position()
            while abs(t_px_pos_x - p_px_pos_x) < block_size * 5 and abs(t_px_pos_y - p_px_pos_y) < block_size * 5:
                t_px_pos_x, t_px_pos_y = random_position()
        for player in range(0, population):
            step = 0
            # Player grid position for learning
            p_px_pos_x, p_px_pos_y = f_px_pos_x, f_px_pos_y
            players[player].pos_x, players[player].pos_y = x_conv(f_px_pos_x), y_conv(f_px_pos_y)
            distance = euclidian_distance(x_conv(t_px_pos_x),
                                          y_conv(t_px_pos_y),
                                          players[player].pos_x,
                                          players[player].pos_y)
            players[player].neural_network.compile('rmsprop')
            running = True
            while running:
                render()

                # Configure inputs of neural network and evaluate outputs
                nn_input = ((players[player].pos_x/20,
                             players[player].pos_y/20,
                             x_conv(t_px_pos_x)/20,
                             y_conv(t_px_pos_y)/20,
                             distance/26.87),)
                nn_output = players[player].neural_network.predict(nn_input)

                # Use output to make a new move
                if np.argmax(nn_output) == 0:
                    new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP})  # create the event
                elif np.argmax(nn_output) == 1:
                    new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN})  # create the event
                elif np.argmax(nn_output) == 2:
                    new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT})  # create the event
                elif np.argmax(nn_output) == 3:
                    new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT})  # create the event
                else:
                    new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE})  # create the event
                pygame.event.post(new_event)  # add the event to the queue

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            if p_px_pos_x >= block_size + left_margin:
                                p_px_pos_x -= block_size
                            else:
                                running = False
                        if event.key == pygame.K_RIGHT:
                            if p_px_pos_x < 640 - block_size + left_margin:
                                p_px_pos_x += block_size
                            else:
                                running = False
                        if event.key == pygame.K_DOWN:
                            if p_px_pos_y < 640 - block_size + up_margin:
                                p_px_pos_y += block_size
                            else:
                                running = False
                        if event.key == pygame.K_UP:
                            if p_px_pos_y >= block_size + up_margin:
                                p_px_pos_y -= block_size
                            else:
                                running = False

                        # Stop conditions
                        if distance == 0:
                            print("GAME WIN - Distance = {}".format(distance))
                            running = False
                        if step == 40:
                            running = False

                        # Update player info
                        step += 1
                        players[player].cur_step = step
                        players[player].pos_x = x_conv(p_px_pos_x)
                        players[player].pos_y = y_conv(p_px_pos_y)
                        distance = euclidian_distance(x_conv(t_px_pos_x),
                                                      y_conv(t_px_pos_y),
                                                      players[player].pos_x,
                                                      players[player].pos_y)
                        players[player].neural_network.fitness_update(distance, step)
                        if players[player].neural_network.fitness > players[player].neural_network.max_fitness:
                            players[player].neural_network.max_fitness = players[player].neural_network.fitness

            # Append player's movements to a list
            players_movement_list.append(players[player].step_list)

        # Log the current generation
        print('Generation: ', generation)

        for player in players:
            pool.append(player.neural_network)

        # Sort based on fitness
        pool = sorted(pool, key=lambda x: x.max_fitness)
        pool.reverse()

        # Find Max Fitness and Log Associated Weights
        for i in range(0, len(pool)):
            # If there is a new max fitness among the population
            if pool[i].max_fitness > max_fitness:
                max_fitness = pool[i].max_fitness
                print('Max Fitness: ', max_fitness)
                pool[i].save_weights('best_nn_weights')

        child = []
        # Crossover between best 4
        for i in range(0, 4):
            for j in range(0, 4):
                # Create a child and add to networks
                child.append(gann.dynamic_crossover(pool[i], pool[j]))

        # Substitute population's neural networks
        for i in range(16):
            players[i].neural_network = child[i]
        for i in range(16, population):
            players[i].neural_network = gann.GeneticANN()

        # Save last generation data
        np.save('data.npy', players_movement_list)
