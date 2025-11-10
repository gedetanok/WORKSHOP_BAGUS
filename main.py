import pygame
from game import Game

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH, HEIGHT = 300, 300
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")

# Main loop
if __name__ == '__main__':
    game = Game(win)
    game.run()