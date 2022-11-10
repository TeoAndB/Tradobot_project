import pygame

class Mouse:
    def __init__(self, screenheight, screenwidth, imagefile):
        self.shape = pygame.image.load(imagefile)

        self.top = screenheight/2 - self.shape.get_height()/2
        self.left = screenwidth/2 - self.shape.get_width()/2

    def Show(self, surface):
        surface.blit(self.shape, (self.left, self.top))

    def UpdateCoords(self, x):
        self.left = x - self.shape.get_width()/2

