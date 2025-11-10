class Board:
    def __init__(self):
        self.board = [None] * 9

    def reset(self):
        self.board = [None] * 9

    def check_winner(self):
        # Check for a winner
        pass