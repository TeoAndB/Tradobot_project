import pygame

WIDTH = 800
HEIGHT = 600
BACKGROUND = (255, 255, 255)

class Ball:
    def __init__(self):
        self.image = pygame.image.load("images/sell_mouse_50x50.png")
        self.speed = [0, 1]
        self.rect = self.image.get_rect()

    def update(self):
        self.move()

    def move(self):
        self.rect = self.rect.move(self.speed)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    ball = Ball()

    while True:
        screen.fill(BACKGROUND)

        point1 = 0, HEIGHT / 2
        point2 = WIDTH, HEIGHT / 2

        pygame.draw.aaline(screen, "black", point1, point2)

        screen.blit(ball.image, ball.rect)
        ball.update()
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()