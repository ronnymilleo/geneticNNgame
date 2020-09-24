import pygame

from gann import GeneticANN
from globals import *


class Player:
    """Player's class"""

    # Constructor. Pass in the color of the block,
    # and its x and y position
    def __init__(self, color, p_width, p_height):
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface([p_width, p_height])
        self.image.fill(color)
        self.image.set_alpha(63)
        self.image.convert_alpha()

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect()

        # User constructor
        self.px_x = 0
        self.px_y = 0
        self.x = 0
        self.y = 0
        self.d = 0
        self.steps = 0
        self.win_status = False
        self.lose_status = False
        self.nn = GeneticANN()
        self.nn_output = 0


def get_player_event(player):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            player.lose_status = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if player.px_x >= b_size + left_margin:
                    player.px_x -= b_size
                else:
                    player.lose_status = True
            if event.key == pygame.K_RIGHT:
                if player.px_x < 640 - b_size + left_margin:
                    player.px_x += b_size
                else:
                    player.lose_status = True
            if event.key == pygame.K_DOWN:
                if player.px_y < 640 - b_size + up_margin:
                    player.px_y += b_size
                else:
                    player.lose_status = True
            if event.key == pygame.K_UP:
                if player.px_y >= b_size + up_margin:
                    player.px_y -= b_size
                else:
                    player.lose_status = True
