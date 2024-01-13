import numpy as np
from tensorflow.keras.models import load_model
import gomoku as gm
import random

def myfun(state):
    path = 'policies/model.h5'
    path_alt = 'model.h5'
    try:
        model = load_model(path)
    except (FileNotFoundError, IOError):
        model = load_model(path_alt)

    valid_actions = state.valid_actions()
    inputO = state.board[gm.MIN]
    inputX = state.board[gm.MAX]
    inp = np.zeros((15, 15),dtype=int)
    inp = np.where(inputO == 1, -1, inp)
    inp = np.where(inputX == 1, 1, inp)
    inp = np.expand_dims(inp, axis=(0, -1))
    tup = calculate(inp,model)
    # Here the model predicts an invalid move
    # After trying multiple implementations to handle this problem, this random function is being used keeping in mind the score vs time tradeoff
    if(tup not in valid_actions):tup = random.choice(valid_actions)
    return tup
def calculate(inp,model):
    output = model.predict(inp).squeeze().reshape((15, 15))
    return np.unravel_index(np.argmax(output), output.shape)
class Submission:
    def __init__(self, board_size, win_size):
        self.win_size = win_size
        self.board_size = board_size

    def __call__(self, state):
        action = myfun(state)
        return action