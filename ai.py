import random

class AI:
    def __init__(self, player):
        self.player = player

    def get_move(self, board):
        available_moves = [i for i in range(9) if board[i] is None]
        return random.choice(available_moves) if available_moves else None