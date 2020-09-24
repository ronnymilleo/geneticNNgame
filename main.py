import math
import os.path
import random

import numpy as np
import pygame

import gann
from globals import *
from player import Player


# Drawing functions
def draw_target(screen, icon):
    screen.blit(icon, (t_px_pos_x, t_px_pos_y))


def draw_flag(screen, icon):
    screen.blit(icon, (f_px_pos_x, f_px_pos_y))


def draw_grid(screen):
    for x in range(left_margin, left_margin + right_margin + b_size, b_size):
        pygame.draw.line(screen, (0, 0, 0), (x, up_margin), (x, height - down_margin), 1)
    for y in range(up_margin, height - down_margin + b_size, b_size):
        pygame.draw.line(screen, (0, 0, 0), (left_margin, y), (left_margin + right_margin, y), 1)


def draw_text(screen, font, generation):
    gen = font.render("Generation = {}".format(generation), True, [0, 0, 0], [255, 255, 255])
    pop = font.render("Population = {}".format(population), True, [0, 0, 0], [255, 255, 255])
    steps = font.render("Steps = {}".format(step), True, [0, 0, 0], [255, 255, 255])
    screen.blits([(gen, (20, 20)), (pop, (20 + 150, 20)), (steps, (20 + 300, 20))])


def draw_fit_table(screen, font, players):
    player_text = font.render("Player", True, [0, 0, 0], [255, 255, 255])
    fitness_text = font.render("Fitness", True, [0, 0, 0], [255, 255, 255])
    steps_text = font.render("Steps", True, [0, 0, 0], [255, 255, 255])
    screen.blits([(player_text, (720, 60)),
                  (fitness_text, (720 + 100, 60)),
                  (steps_text, (720 + 200, 60))])
    for p in range(0, len(players)):
        player_id = font.render("P{}".format(p), True, [0, 0, 0], [255, 255, 255])
        fitness_number = font.render("{:.4f}".format(players[p].nn.max_fitness), True, [0, 0, 0],
                                     [255, 255, 255])
        steps_number = font.render("{:02d}".format(players[p].steps), True, [0, 0, 0], [255, 255, 255])
        screen.blits([(player_id, (720, 60 + p * 24 + 24)),
                      (fitness_number, (720 + 100, 60 + p * 24 + 24)),
                      (steps_number, (720 + 200, 60 + p * 24 + 24))])


def draw_players(screen, players):
    player_list = []
    for player in players:
        player_list.append((player.image, (player.px_x, player.px_y)))
    screen.blits(player_list)


def render(screen, font, f_icon, t_icon, generation, players):
    draw_text(screen, font, generation)
    draw_grid(screen)
    draw_flag(screen, f_icon)
    draw_target(screen, t_icon)
    draw_fit_table(screen, font, players)
    draw_players(screen, players)
    pygame.display.flip()


# Calculations and converting functions
def euclidian_distance(x1, y1, x2, y2):
    new_distance = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return new_distance


def x_conv(px_x):
    return (px_x - left_margin) // 32 + 1


def y_conv(px_y):
    return 20 - (px_y - up_margin) // 32


def random_position():
    return math.floor(random.random() * (640 / b_size)) * b_size + left_margin, \
           math.floor(random.random() * (640 / b_size)) * b_size + up_margin


def ai_move(ai):
    # Configure inputs of neural network and evaluate outputs
    nn_input = ((ai.x / 20,
                 ai.y / 20,
                 x_conv(t_px_pos_x) / 20,
                 y_conv(t_px_pos_y) / 20,
                 ai.d / 26.87),)
    ai.nn_output = ai.nn.predict(nn_input)
    if np.argmax(ai.nn_output) == 0:
        if ai.px_x >= b_size + left_margin:
            ai.px_x -= b_size
        else:
            ai.lose_status = True
    elif np.argmax(ai.nn_output) == 1:
        if ai.px_x < 640 - b_size + left_margin:
            ai.px_x += b_size
        else:
            ai.lose_status = True
    elif np.argmax(ai.nn_output) == 2:
        if ai.px_y < 640 - b_size + up_margin:
            ai.px_y += b_size
        else:
            ai.lose_status = True
    else:
        if ai.px_y >= b_size + up_margin:
            ai.px_y -= b_size
        else:
            ai.lose_status = True


def ai_update(ai):
    # Update distance based on movement
    ai.d = euclidian_distance(x_conv(t_px_pos_x),
                              y_conv(t_px_pos_y),
                              ai.x,
                              ai.y)
    # Update player info every round
    ai.x = x_conv(ai.px_x)
    ai.y = y_conv(ai.px_y)
    # Update rect position
    ai.rect = (ai.px_x, ai.px_y)
    # Update fitness every round
    ai.nn.fitness_update(ai.d)
    # Save max fitness info
    if ai.nn.fitness > ai.nn.max_fitness:
        ai.nn.max_fitness = ai.nn.fitness
    # New step
    ai.steps = step


def reset_players(players_array):
    # Update player's pixel position, distance to target and status
    for player in players_array:
        player.lose_status = False
        player.win_status = False
        player.x, player.y = x_conv(f_px_pos_x), y_conv(f_px_pos_y)
        player.px_x, player.px_y = f_px_pos_x, f_px_pos_y
        player.rect = (f_px_pos_x, f_px_pos_y)
        player.d = euclidian_distance(x_conv(t_px_pos_x),
                                      y_conv(t_px_pos_y),
                                      player.x,
                                      player.y)


def randomize_objectives():
    global f_px_pos_x, f_px_pos_y, t_px_pos_x, t_px_pos_y
    # Generate new random positions for start and target
    f_px_pos_x, f_px_pos_y = random_position()
    t_px_pos_x, t_px_pos_y = random_position()
    # Check if target is at same position as start flag
    while abs(t_px_pos_x - f_px_pos_x) < b_size * 5 and abs(t_px_pos_y - f_px_pos_y) < b_size * 5:
        t_px_pos_x, t_px_pos_y = random_position()


def update_generation(generation, players):
    # Log the current generation
    print('Generation: ', generation)

    pool = []
    for player in players:
        pool.append(player.nn)

    # Sort based on fitness
    pool = sorted(pool, key=lambda x: x.mean_fitness)
    pool.reverse()

    # Find Max Fitness and Log Associated Weights
    for i in range(0, len(pool)):
        # If there is a new max fitness among the population
        if pool[i].max_fitness > max_fitness:
            pool[i].save_weights('best_nn_weights')

    child = []
    # Crossover between best
    for i in range(0, 5):
        for j in range(5, 10):
            # Create a child and add to networks
            if i != j:
                child.append(gann.dynamic_crossover(pool[i], pool[j]))

    # Substitute population's neural networks
    for i in range(population):
        players[i].nn = child[i]


def main():
    global step
    # Initialise screen
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('PYGAME AI')

    # Choose icons
    game_icon = pygame.image.load('deep-learning.png').convert_alpha()
    flag_icon = pygame.image.load('start_flag.png').convert_alpha()
    target_icon = pygame.image.load('target.png').convert_alpha()
    pygame.display.set_icon(game_icon)

    # Fill game background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((255, 255, 255))

    # Font
    font = pygame.font.Font(os.path.curdir + '\\font\\Roboto-Regular.ttf', 20)

    # Generate population
    players = []

    # First time
    randomize_objectives()

    # The first distance is calculated by the Player's constructor
    for i in range(0, population):
        players.append(Player(color=(0, 0, 127), p_width=b_size, p_height=b_size))

    # Compile every neural network
    for player in players:
        player.nn.compile()

    # Generations loop
    for generation in range(0, generations):
        # Every generation clear screen and update
        screen.blit(background, (0, 0))
        pygame.display.flip()
        # Every N generations, change the start and target's position
        div, res = divmod(generation, 10)
        if res == 0:
            randomize_objectives()
        reset_players(players)

        # Game loop
        running = True
        step = 0
        fitness_per_round = np.zeros(population)
        while running:
            pygame.event.get()
            render(screen, font, flag_icon, target_icon, generation, players)
            # Each for loop is 1 movement for every player
            for p in range(0, population):
                # End condition
                if players[p].lose_status or players[p].win_status:
                    print(fitness_per_round[p])
                    continue
                else:
                    ai_move(players[p])
                    ai_update(players[p])
                    fitness_per_round[p] = players[p].nn.fitness

            step += 1
            if step == 40:
                running = False

        update_generation(generation, players)


if __name__ == '__main__':
    f_px_pos_x = 0
    f_px_pos_y = 0
    t_px_pos_x = 0
    t_px_pos_y = 0
    max_fitness = 0
    step = 0
    main()
