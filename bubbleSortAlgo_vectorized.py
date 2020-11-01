import time
import pickle
from collections import Counter, deque
import csv
from tqdm import tqdm
from itertools import product
import sys
import numpy as np

def getTop(state, i):
    if state[i,3]:
        return state[i,3]

    if state[i,2]:
        return state[i,2]

    if state[i,1]:
        return state[i,1]

    return state[i,0]

def setTop(state, i, ball):
    if not state[i,0]:
        state[i,0] = ball
    elif not state[i,1]:
        state[i,1] = ball
    elif not state[i,2]:
        state[i,2] = ball
    else:
        state[i,3] = ball

def remTop(state, i):
    if state[i,3]:
        state[i,3] = 0
    elif state[i,2]:
        state[i,2] = 0
    elif state[i,1]:
        state[i,1] = 0
    else:
        state[i,0] = 0

def isEmpty(state, i):
    return state[i,0] == 0

def isFull(state, i):
    return state[i,3] != 0

def moveBall(state, i, j):
    ball = getTop(state, i)
    setTop(state, j, ball)
    remTop(state, i)

def getReachableStates(state):
    np.argwhere(state)
            moveBall(state, i, j)
            yield state
            moveBall(state, j, i)

def isSolved(state):
    for i in range(state.shape[0]):
        if not np.all(state[i,0] == state[i,:]):
            return False
    return True

char_to_uint = lambda c: {
    ' ': 0,
    'p': 1,
    'r': 2,
    'l': 3,
    'y': 4,
    'o': 5,
    'b': 6,
    'w': 7,
    'i': 8,
    't': 9,
    'g': 10,
    'e': 11,
    'd': 12,
}[c]

"""
p - purple
r - red
l - light blue
y - yellow
o - orange
b - blue
w - brown
i - pink
t - teal
g - gray
e - green
d - dark yellow
"""

INITIAL_STATE = 'prly' + 'oyoy' + 'bwpi' + 'bgtb' + 'ewei' + 'ypgr' + 'digr' + 'dpgl' + 'teot' + 'owll' + 'tdeb' + 'ridw' + '    ' + '    '

assert all([(n == 4) for c, n in Counter(INITIAL_STATE).items() if c != ' '])

INITIAL_STATE = np.array(list(map(char_to_uint, INITIAL_STATE)), dtype=np.uint8).reshape((len(INITIAL_STATE) // 4, 4), order='C')


def search(state_str, depth, seen_str, sols, max_depth, num_bins):
    
    if depth > max_depth:
        return

    if state_str in seen_str:
        return

    seen_str[state_str] = True
    
    state_np = np.fromiter(
        state_str,
        dtype=np.uint8,
        count=num_bins * 4,
    ).reshape(
        (num_bins,4),
    )

    if isSolved(state_np):
        sols.append(state_np)
        print(f'found sol at depth {depth}')
        return

    for adj_state_np in getReachableStates(state_np):
        adj_state_str = adj_state_np.tostring()
        search(adj_state_str, depth + 1, seen_str, sols, max_search_depth, num_bins)


if __name__ == '__main__':
    START_TIME = time.time()

    seen_str = {}
    sols = []
    max_search_depth = 40
    num_bins = INITIAL_STATE.shape[0]
    state_str = INITIAL_STATE.tostring()

    search(state_str, 0, seen_str, sols, max_search_depth, num_bins)

    END_TIME = time.time()

    print(f'searched {len(seen_str)} unique states to a max depth of {max_search_depth} in {round(END_TIME-START_TIME, 3)} seconds')
    print(f'found {len(sols)} unique solutions')
    #print(f'shortest sol was {min([rank[s] for s in sols], default=None)}')
