import pygame
# Import pygame.locals for easier access to key coordinates
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
)

from gann import GeneticANN
from globals import *


class Player(pygame.sprite.Sprite):
    """Player's class"""

    def __init__(self):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        self.image = pygame.Surface((b_size, b_size))
        self.image.fill(PLAYER_BLUE)
        self.image.set_alpha(63)
        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect(left=left_margin, top=top_margin)
        # Player status
        self.win_status = False
        self.lose_status = False
        # Neural Network inputs
        self.x = 0
        self.y = 0
        self.distance = 0
        self.quadrant = 0
        self.x_angle = 0
        self.y_angle = 0
        # Neural Network
        self.nn = GeneticANN()
        self.nn_output = 0
        self.steps = 0

    def move(self, pressed_keys):
        movement = False
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -b_size)
            movement = True
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, b_size)
            movement = True
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-b_size, 0)
            movement = True
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(b_size, 0)
            movement = True

        # Keep player on the screen
        if self.rect.left <= left_margin:
            self.rect.left = left_margin
        if self.rect.right >= b_size * 20 + left_margin:
            self.rect.right = b_size * 20 + left_margin
        if self.rect.top <= top_margin:
            self.rect.top = top_margin
        if self.rect.bottom >= b_size * 20 + top_margin:
            self.rect.bottom = b_size * 20 + top_margin

        return movement
