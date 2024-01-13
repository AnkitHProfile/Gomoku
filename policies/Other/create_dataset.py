import numpy as np
import os
from glob import glob
from tqdm import tqdm

base_path = '/Users/Aditya Porwal/Downloads/gomocup2019results'
output_path = os.path.join('dataset', os.path.basename(base_path))
os.makedirs(output_path, exist_ok=True)
file_list = glob(os.path.join(base_path, '%s*/*.psq' % ('Standard', )))

for index, file_path in enumerate(tqdm(file_list)):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines() 

    w, h = lines[0].split(' ')[1].strip(',').split('x')
    w, h = int(w), int(h)

    lines = lines[1:]

    inputs, outputs = [], []
    board = np.zeros([h, w], dtype=np.int8)

    for i, line in enumerate(lines):
        if ',' not in line:
            break

        x, y, t = np.array(line.split(','), np.int8)

        if i % 2 == 0:
            player = 1
        else:
            player = 2

        input = board.copy().astype(np.int8)
        input[(input != player) & (input != 0)] = -1
        input[(input == player) & (input != 0)] = 1

        output = np.zeros([h, w], dtype=np.int8)
        output[y-1, x-1] = 1

        for k in range(4):
            input_r = np.rot90(input, k=k)
            output_r = np.rot90(output, k=k)
            inputs.append(input_r)
            outputs.append(output_r)
            inputs.append(np.fliplr(input_r))
            outputs.append(np.fliplr(output_r))
            inputs.append(np.flipud(input_r))
            outputs.append(np.flipud(output_r))

        board[y-1, x-1] = player

    np.savez_compressed(os.path.join(output_path, '%s.npz' % (str(index).zfill(5))), inputs=inputs, outputs=outputs)