import pytest
from game import Game


def test_game_initialization():
    game = Game(None)
    assert game.current_player == 'X'
    assert game.board == [None] * 9


def test_game_run():
    game = Game(None)
    game.run()  # This should run without errors


def test_ai_move():
    from ai import AI
    ai = AI('O')
    board = [None] * 9
    move = ai.get_move(board)
    assert move in range(9)


def test_board_reset():
    from board import Board
    board = Board()
    board.reset()
    assert board.board == [None] * 9


def test_winner_check():
    from board import Board
    board = Board()
    board.board = ['X', 'X', 'X', None, None, None, None, None, None]
    assert board.check_winner() == 'X'  # Assuming check_winner() is implemented correctly