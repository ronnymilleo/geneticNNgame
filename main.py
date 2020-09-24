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
    for x in range(left_margin, left_margin + right_margin + b_size, b_size):
        pygame.draw.line(screen, (0, 0, 0), (x, up_margin), (x, height - down_margin), 1)
    for y in range(up_margin, height - down_margin + b_size, b_size):
        pygame.draw.line(screen, (0, 0, 0), (left_margin, y), (left_margin + right_margin, y), 1)


def draw_text():
    global step
    gen = font.render("Generation = {}".format(generation), True, [0, 0, 0], [255, 255, 255])
    pop = font.render("Population = {}".format(population), True, [0, 0, 0], [255, 255, 255])
    steps = font.render("Steps = {}".format(step), True, [0, 0, 0], [255, 255, 255])
    # dist = font.render("Distance = {:.2f}".format(player.distance), True, [0, 0, 0], [255, 255, 255])
    # position = font.render("Position = ({}, {})".format(player.pos_x, player.pos_y), True, [0, 0, 0],
    #                        [255, 255, 255])
    screen.blit(gen, (20, 20))
    screen.blit(pop, (20 + 150, 20))
    screen.blit(steps, (20 + 300, 20))
    # screen.blit(dist, (20 + 400, 20))
    # screen.blit(position, (20 + 550, 20))


def draw_fit_table(player_array):
    global step
    player_text = font.render("Player", True, [0, 0, 0], [255, 255, 255])
    fitness_text = font.render("Fitness", True, [0, 0, 0], [255, 255, 255])
    screen.blit(player_text, (720, 60))
    screen.blit(fitness_text, (720 + 100, 60))
    for p in range(0, len(player_array)):
        player_id = font.render("P{}".format(p), True, [0, 0, 0], [255, 255, 255])
        fitness_number = font.render("{:.4f}".format(player_array[p].neural_network.max_fitness), True, [0, 0, 0],
                                     [255, 255, 255])
        steps_number = font.render("{}".format(player_array[p].steps), True, [0, 0, 0], [255, 255, 255])
        screen.blit(player_id, (720, 60 + p * 24 + 24))
        screen.blit(fitness_number, (720 + 100, 60 + p * 24 + 24))
        screen.blit(steps_number, (720 + 200, 60 + p * 24 + 24))


def plot(data):
    ax.plot(data)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    return pygame.image.fromstring(raw_data, size, "RGB")


def draw_player(rect, pos_x, pos_y):
    screen.blit(rect, (pos_x, pos_y))


def render():
    screen.fill((255, 255, 255), text_background)
    draw_text()
    draw_grid()
    draw_flag()
    draw_target()
    # screen.blit(surf, (720 + 200, 60))
    pygame.display.update()
    clock.tick(30)


# Calculations and converting functions
def euclidian_distance(dest_pos_x, dest_pos_y, origin_pos_x, origin_pos_y):
    new_distance = math.sqrt(math.pow(dest_pos_x - origin_pos_x, 2) + math.pow(dest_pos_y - origin_pos_y, 2))
    return new_distance


def x_conv(px_x):
    return (px_x - left_margin) // 32 + 1


def y_conv(px_y):
    return 20 - (px_y - up_margin) // 32


def random_position():
    return math.floor(random.random() * (640 / b_size)) * b_size + left_margin, \
           math.floor(random.random() * (640 / b_size)) * b_size + up_margin


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
    b_size = 32
    left_margin = 40
    right_margin = b_size * 20
    up_margin = 60
    down_margin = 20
    text_background = pygame.Rect(720, 0, 720, 720)

    # Game display
    screen = pygame.display.set_mode((width, height))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Mistery's game")
    icon = pygame.image.load('deep-learning.png')
    pygame.display.set_icon(icon)
    flag_icon = pygame.image.load('start_flag.png')
    target_icon = pygame.image.load('target.png')
    font = pygame.font.Font(os.path.curdir + '\\font\\Roboto-Regular.ttf', 20)

    # Max generations and initial population
    generations = 100
    population = 20
    max_fitness = 0
    step = 0
    # Generate n randomly weighted neural networks
    # Create a List of all active GeneticNeuralNetworks
    players = []
    # Save movements for later
    players_movement_list = []
    # Aux pool to make crossover and mutation
    pool = []

    # Generate start position for every player based on flag position
    f_px_pos_x, f_px_pos_y = random_position()
    # Generate target position
    t_px_pos_x, t_px_pos_y = random_position()
    # The first distance is calculated by the constructor
    # Player class algo has the information of pixel position for the player
    for i in range(0, population):
        players.append(Player(f_px_pos_x, f_px_pos_y, x_conv(t_px_pos_x), y_conv(t_px_pos_y)))

    # Check if target is at same position as start flag
    while abs(t_px_pos_x - players[0].px_pos_x) < b_size * 5 and abs(t_px_pos_y - players[0].px_pos_y) < b_size * 5:
        t_px_pos_x, t_px_pos_y = random_position()

    # Compile every neural network
    for player in players:
        player.neural_network.compile()

    # Main loop
    for generation in range(0, generations):
        # Every generation clear screen
        screen.fill((255, 255, 255))
        pygame.display.update()
        clock.tick(30)
        # Every 10 generations, change the start and target's position
        div, res = divmod(generation, 5)
        if res == 0:
            # Update flag and player's start position
            f_px_pos_x, f_px_pos_y = random_position()
            t_px_pos_x, t_px_pos_y = random_position()
            while abs(t_px_pos_x - players[0].px_pos_x) < b_size * 5 and \
                abs(t_px_pos_y - players[0].px_pos_y) < b_size * 5:
                t_px_pos_x, t_px_pos_y = random_position()

        # Update player's pixel position, distance to target and status
        for player in players:
            player.lose_status = False
            player.win_status = False
            player.px_pos_x, player.px_pos_y = f_px_pos_x, f_px_pos_y
            player.distance = euclidian_distance(x_conv(t_px_pos_x),
                                                 y_conv(t_px_pos_y),
                                                 player.pos_x,
                                                 player.pos_y)

        running = True
        step = 0
        stop_counter = 0
        while running:
            pygame.event.get()
            render()
            for player in players:
                if player.lose_status or player.win_status:
                    continue
                else:
                    draw_player(player.p_rect, player.px_pos_x, player.px_pos_y)
                    draw_fit_table(players)
                    pygame.display.update()
                    clock.tick(30)

                # Configure inputs of neural network and evaluate outputs
                nn_input = ((player.pos_x / 20,
                             player.pos_y / 20,
                             x_conv(t_px_pos_x) / 20,
                             y_conv(t_px_pos_y) / 20,
                             player.distance / 26.87),)
                player.nn_output = player.neural_network.predict(nn_input)

                # Use output to make a new move creating a new keyboard event
                if np.argmax(player.nn_output) == 0:
                    # new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT})
                    if player.px_pos_x >= b_size + left_margin:
                        player.px_pos_x -= b_size
                    else:
                        player.lose_status = True
                elif np.argmax(player.nn_output) == 1:
                    # new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT})
                    if player.px_pos_x < 640 - b_size + left_margin:
                        player.px_pos_x += b_size
                    else:
                        player.lose_status = True
                elif np.argmax(player.nn_output) == 2:
                    # new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN})
                    if player.px_pos_y < 640 - b_size + up_margin:
                        player.px_pos_y += b_size
                    else:
                        player.lose_status = True
                elif np.argmax(player.nn_output) == 3:
                    # new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP})
                    if player.px_pos_y >= b_size + up_margin:
                        player.px_pos_y -= b_size
                    else:
                        player.lose_status = True
                else:
                    # new_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE})
                    pass

                # Add the event to the queue
                # pygame.event.post(new_event)

                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         player.lose_status = True
                #     if event.type == pygame.KEYDOWN:
                #         if event.key == pygame.K_LEFT:
                #             if player.px_pos_x >= b_size + left_margin:
                #                 player.px_pos_x -= b_size
                #             else:
                #                 player.lose_status = True
                #         if event.key == pygame.K_RIGHT:
                #             if player.px_pos_x < 640 - b_size + left_margin:
                #                 player.px_pos_x += b_size
                #             else:
                #                 player.lose_status = True
                #         if event.key == pygame.K_DOWN:
                #             if player.px_pos_y < 640 - b_size + up_margin:
                #                 player.px_pos_y += b_size
                #             else:
                #                 player.lose_status = True
                #         if event.key == pygame.K_UP:
                #             if player.px_pos_y >= b_size + up_margin:
                #                 player.px_pos_y -= b_size
                #             else:
                #                 player.lose_status = True

                # Update distance based on movement
                player.distance = euclidian_distance(x_conv(t_px_pos_x),
                                                     y_conv(t_px_pos_y),
                                                     player.pos_x,
                                                     player.pos_y)

                # Stop conditions
                if player.distance == 0:
                    print("GAME WIN - Distance = {}".format(player.distance))
                    player.win_status = True

                # Update player info every round
                player.pos_x = x_conv(player.px_pos_x)
                player.pos_y = y_conv(player.px_pos_y)

                # Update fitness every round
                player.neural_network.fitness_update(player.distance)
                # Save max fitness info
                if player.neural_network.fitness > player.neural_network.max_fitness:
                    player.neural_network.max_fitness = player.neural_network.fitness

                player.steps = step
            step += 1
            if step == 40:
                running = False

        # Log the current generation
        print('Generation: ', generation)

        pool = []
        for player in players:
            pool.append(player.neural_network)

        # Sort based on fitness
        pool = sorted(pool, key=lambda x: x.max_fitness)
        pool.reverse()

        # Find Max Fitness and Log Associated Weights
        for i in range(0, len(pool)):
            # If there is a new max fitness among the population
            if pool[i].max_fitness > max_fitness:
                pool[i].save_weights('best_nn_weights')

        child = []
        # Crossover between best 4
        for i in range(0, 5):
            for j in range(5, 10):
                # Create a child and add to networks
                if i != j:
                    child.append(gann.dynamic_crossover(pool[i], pool[j]))

        # Substitute population's neural networks
        for i in range(population):
            players[i].neural_network = child[i]

        # Save last generation data
        np.save('data.npy', players_movement_list)
