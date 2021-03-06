import time
from collections import Counter, deque
from itertools import product
import pickle
import numpy as np
from heapq import heappush, heappop
import sys
from image import ImageLayer

def moveBall(state, i, j):
    # get raw index of top ball in bin i
    raw_i = 4 * i + 3
    if state[raw_i] != ' ':
        pass
    elif state[raw_i - 1] != ' ':
        raw_i -= 1
    elif state[raw_i - 2] != ' ':
        raw_i -= 2
    else:
        raw_i -= 3
    
    # get raw index of first empty space in bin j
    raw_j = 4 * j
    if state[raw_j] == ' ':
        pass
    elif state[raw_j + 1] == ' ':
        raw_j += 1
    elif state[raw_j + 2] == ' ':
        raw_j += 2
    else:
        raw_j += 3
    
    if raw_i < raw_j:
        newState = ''.join((state[:raw_i],          ' ', state[raw_i + 1:raw_j], state[raw_i], state[raw_j+1:]))
    else:
        newState = ''.join((state[:raw_j], state[raw_i], state[raw_j + 1:raw_i],          ' ', state[raw_i+1:]))

    return newState

def getCaches(state, numBins):
    binHeight,topBall,valid_i,valid_j = [None] * numBins,[None] * numBins,[],[]

    for i in range(numBins):

        raw_i = 4 * i

        if state[raw_i + 3] != ' ':
            topBall[i] = state[raw_i + 3]
            binHeight[i] = 4
            valid_i.append(i)

        elif state[raw_i + 2] != ' ':
            topBall[i] = state[raw_i + 2]
            binHeight[i] = 3
            valid_i.append(i)
            valid_j.append(i)

        elif state[raw_i + 1] != ' ':
            topBall[i] = state[raw_i + 1]
            binHeight[i] = 2
            valid_i.append(i)
            valid_j.append(i)

        elif state[raw_i    ] != ' ':
            topBall[i] = state[raw_i    ]
            binHeight[i] = 1
            valid_i.append(i)
            valid_j.append(i)

        else:
            topBall[i] = ' '
            binHeight[i] = 0
            valid_j.append(i)
    
    return binHeight,topBall,valid_i,valid_j

def getReachableStates(state):
    numBins = len(state) // 4

    # create bin heigh and top ball caches
    binHeight,topBall,valid_i,valid_j = getCaches(state, numBins)

    reachable_states = []

    # search through all possible moves and yield neighbors
    for i in valid_i:

        if state[4 * i] == state[4 * i + 1] == state[4 * i + 2] == state[4 * i + 3]: # bin is completed, don't touch
            continue
            
        for j in valid_j:

            if i == j:
                continue

            if state[4 * j] == state[4 * j + 1] == state[4 * j + 2] == topBall[i]:
                return [moveBall(state, i, j)]
            
            if binHeight[j] == 0 or topBall[i] == topBall[j]:
                reachable_states.append(moveBall(state, i, j))
    
    return reachable_states

def isSolved(state):
    numBins = len(state) // 4

    for i in range(numBins):
        if not (state[4 * i] == state[4 * i + 1] == state[4 * i + 2] == state[4 * i + 3]):
            return False
    return True

def cost(state):
    numBins = len(state) // 4
    return numBins - 2 - sum((state[4 * i] == state[4 * i + 1] == state[4 * i + 2] == state[4 * i + 3] != ' ') for i in range(numBins))

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
n - indigo
"""

def search(initial_state, max_depth, sols_before_return):

    work_queue = [(cost(initial_state), 0, initial_state)]
    seen = {}
    pred_dict = {}
    sol = ''
    sol_depth = max_depth + 1

    seen[initial_state] = 0
    pred_dict[initial_state] = None

    incr = 10
    l_time = time.time()
    s_time = l_time

    while len(work_queue):

        if time.time() > incr + l_time:
            l_time = time.time()
            print(f'work_queue_size={len(work_queue)}  nodes_searched={len(seen)}  time_elapsed={int(l_time-s_time)} seconds')

        c, depth, state = heappop(work_queue)

        if state in seen and seen[state] < depth:
            continue

        if isSolved(state):
            sols_before_return -= 1
            if depth < sol_depth:
                sol = state
                sol_depth = depth
                max_depth = depth
                print(f'found new sol at depth {depth}')

            if sols_before_return == 0:
                break
        
        if depth + 1 < max_depth:
            for neighbor_state in getReachableStates(state):
                if neighbor_state not in seen or seen[neighbor_state] > depth + 1:
                    seen[neighbor_state] = depth + 1
                    pred_dict[neighbor_state] = state

                    heappush(work_queue, (cost(neighbor_state), depth + 1, neighbor_state))

    return sol, sol_depth, pred_dict, len(seen)


if __name__ == '__main__':

    imageLayer = ImageLayer('./input/', './output/')

    if len(sys.argv) != 3:
        print('incorrect syntax for cmd')
        exit()
    
    filename = str(sys.argv[1])
    num_bins = int(sys.argv[2])

    if num_bins < 0:
        print('generating video from pre-existing solution...')
        path = pickle.load(open(filename,'rb'))
        imageLayer.createSolPathVideo(filename.split('.')[0], path)
        exit()


    initial_state = imageLayer.parseImage(filename, num_bins)

    assert all([(n == 4) for c, n in Counter(initial_state).items() if c != ' '])

    max_search_depth = 75 # start high, quasi dfs finds first solution quickly and lowers this value to best depth of sol
    sols_before_return = -1 # exhaustive

    START_TIME = time.time()
    sol, depth, pred_dict, nodes_searched = search(initial_state, max_search_depth, sols_before_return)
    END_TIME = time.time()

    print(f'searched {nodes_searched} unique states to a max depth of {max_search_depth} in {round(END_TIME-START_TIME, 3)} seconds')

    if sol:
        print(f'sol was {depth} steps long')
        
        s = sol
        path = []

        while s:
            path.append(s)
            s = pred_dict[s]
        
        path.reverse()

        pickle.dump(path, open(filename.split('.')[0] + '_solution.p', 'wb'))
        
        imageLayer.createSolPathVideo(filename.split('.')[0], path)
