from gann import GeneticANN
import math
import pygame


class Player:
    """Player's class"""
    p_rect = pygame.Surface((32, 32))
    p_rect.fill((0, 0, 127))
    p_rect.set_alpha(63)

    def __init__(self, p_px_pos_x, p_px_pos_y, tg_x, tg_y):
        self.px_pos_x = p_px_pos_x
        self.px_pos_y = p_px_pos_y
        self.pos_x = (self.px_pos_x - 40) // 32 + 1
        self.pos_y = 20 - (self.px_pos_y - 60) // 32
        self.distance = math.sqrt(math.pow(self.pos_x - tg_x, 2) + math.pow(self.pos_y - tg_y, 2))
        self.steps = 0
        self.win_status = False
        self.lose_status = False
        self.neural_network = GeneticANN()
        self.nn_output = 0
