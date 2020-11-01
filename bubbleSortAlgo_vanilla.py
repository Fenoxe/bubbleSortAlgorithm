import time
import pickle
from collections import Counter, deque
import csv
from tqdm import tqdm
from itertools import product
import sys

def getTop(state, i):
    if state[4 * i + 3] != ' ':
        return state[4 * i + 3]
    if state[4 * i + 2] != ' ':
        return state[4 * i + 2]
    if state[4 * i + 1] != ' ':
        return state[4 * i + 1]
    return state[4 * i]

def setTop(state, i, ball):
    if state[4 * i] == ' ':
        return ''.join((state[:4 * i],ball,state[4 * i + 1:]))
    if state[4 * i + 1] == ' ':
        return ''.join((state[:4 * i + 1],ball,state[4 * i + 2:]))
    if state[4 * i + 2] == ' ':
        return ''.join((state[:4 * i + 2],ball,state[4 * i + 3:]))
    return ''.join((state[:4 * i + 3],ball,state[4 * i + 4:]))

def remTop(state, i):
    if state[4 * i + 3] != ' ':
        return ''.join((state[:4 * i + 3],' ',state[4 * i + 4:]))
    if state[4 * i + 2] != ' ':
        return ''.join((state[:4 * i + 2],' ',state[4 * i + 3:]))
    if state[4 * i + 1] != ' ':
        return ''.join((state[:4 * i + 1],' ',state[4 * i + 2:]))
    return ''.join((state[:4 * i],' ',state[4 * i + 1:]))

def isEmpty(state, i):
    return state[4 * i] == ' '

def isFull(state, i):
    return state[4 * i + 3] != ' '

def moveBall(state, i, j):
    ball = getTop(state, i)
    newState = setTop(state, j, ball)
    newState = remTop(newState, i)
    return newState

def getReachableStates(currState):
    numBins = len(currState) // 4

    empty_bins = set( i for i in range(numBins) if isEmpty(currState, i) )
    full_bins  = set( i for i in range(numBins) if isFull(currState, i) )
    top_ball = [ getTop(currState, i) for i in range(numBins) ]
    
    for i,j in product(range(numBins), range(numBins)):
        if i == j:
            continue
        if i in empty_bins:
            continue
        if j in full_bins:
            continue
        if j in empty_bins or (top_ball[i] == top_ball[j]):
            yield moveBall(currState, i, j)

def isSolved(state):
    numBins = len(state) // 4

    for i in range(numBins):
        bin_t = state[4 * i : 4 * i + 4]
        if bin_t != 4 * bin_t[0]:
            return False
    return True

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


def search(state, from_state, depth, rank, prev, sols, max_depth):

    if depth > max_depth:
        return

    if state in rank:
        if rank[state] > depth:
            rank[state] = depth
            prev[state] = from_state
        
        return
    
    rank[state] = depth
    prev[state] = from_state

    if isSolved(state):
        sols.append(state)
        print(f'found sol at depth {depth}')
        return

    for neighbor_state in getReachableStates(state):
        search(neighbor_state, state, depth + 1, rank, prev, sols, max_search_depth)



if __name__ == '__main__':
    rank = {}
    prev = {}
    sols = []
    max_search_depth = 40

    START_TIME = time.time()

    search(INITIAL_STATE, None, 0, rank, prev, sols, max_search_depth)

    END_TIME = time.time()

    print(f'searched {len(rank)} unique states to a max depth of {max_search_depth} in {round(END_TIME-START_TIME, 3)} seconds')
    print(f'found {len(sols)} unique solutions')

    best_sol, best_rank = min([(s, rank[s]) for s in sols], key=lambda a: a[1], default=(None, -1))
    print(f'shortest sol was {best_rank} steps long')
    
    s = best_sol
    path = []

    while s:
        path.append(s)
        s = prev[s]
    
    path.reverse()

    for i,s in enumerate(path):
        i_str = ' ' * (2 - len(str(i))) + str(i) + ' | '
        s_str = ' '.join([s[4*i : 4*(i+1)] for i in range(len(s) // 4)])

        print(i_str + s_str)

