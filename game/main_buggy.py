import pygame
import sys
import os
from pygame.locals import *
from Mouse import *
from Orb import *

# TODO: Add a line
# TODO: Add a moving ball according to SP500 coordinates
folder = 'images'

pygame.init()

clock = pygame.time.Clock()
screenheight = 600
screenwidth = 1000

screen = pygame.display.set_mode((screenwidth, screenheight))

bg = pygame.image.load(os.path.join("./", f'{folder}/background_white.png'))

pygame.mouse.set_visible(0)
pygame.display.set_caption('Tradobot Game')


Hero = Mouse(screenheight, screenwidth, f'{folder}/sell_mouse_50x50.png')
ball = Ball()

dummy_time = [0,20,30]
dummy_list = [500,700,800]

while True:
    framerate = 60
    clock.tick(framerate)

    screen.blit(ball.image, ball.rect)
    ball.update()

    screen.blit(bg, (0, 0))
    x,y = pygame.mouse.get_pos()

    Hero.UpdateCoords(x)
    point1 = 0, screenheight/2 - Hero.shape.get_height()/5
    point2 = screenwidth, screenheight/2 - Hero.shape.get_height()/5

    pygame.draw.aaline(screen,"black", point1, point2)

    # set a dot in the middle of the line: intial position: x,y
    # calculate step based on the list: dx, dy

    Hero.Show(screen)
    #ball.update()



    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()

    pygame.display.update()