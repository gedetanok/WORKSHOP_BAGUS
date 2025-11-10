import pygame

class Game:
    def __init__(self, win):
        self.win = win
        self.board = [None] * 9
        self.current_player = 'X'

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.draw()

    def draw(self):
        self.win.fill((255, 255, 255))
        # Draw the board and other elements
        pygame.display.update()