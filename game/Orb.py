import pygame

class Ball:
    def __init__(self):
        self.image = pygame.image.load("images/sell_mouse_50x50.png")
        self.speed = [0, 1]
        self.rect = self.image.get_rect()

    def update(self):
        self.move()

    def move(self):
        self.rect = self.rect.move(self.speed)