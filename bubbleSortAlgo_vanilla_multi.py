import time
import pickle
from collections import Counter, deque
import csv
from tqdm import tqdm
from itertools import product
import sys
from multiprocessing import Process, Queue, Manager, cpu_count
import os

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

    empty_bins = set( i for i in range(numBins) if isEmpty(currState, i))
    full_bins  = set( i for i in range(numBins) if isFull(currState, i))
    top_ball = [getTop(currState, i) for i in range(numBins)]

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
        bin_t = state[4 * i:4 * i + 4]
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


def search_worker(work_q, rank_d, max_depth, sols, num_sols_found, manager):

    ran_once = False

    while not work_q.empty():
        ran_once = True

        try:
            state, depth = work_q.get(timeout=1)
        except:
            continue
        
        if depth > max_depth:
            continue

        if state in rank_d:
            continue

        rank_d[state] = manager.list()

        if isSolved(state):
            num_sols_found[0] += 1
            print(f'found sol at depth {depth}\n', end='')
            sols.append(state)
            continue

        for neighbor_state in getReachableStates(state):
            rank_d[state].append(neighbor_state)
            work_q.put((neighbor_state, depth + 1), timeout=1)
    
    suffix = '' if ran_once else '(did not run)'
    print(f'process {os.getpid()} has quit {suffix}\n', end='')

def search(state, depth, rank, sols, max_depth):

    if depth > max_depth:
        return 0

    if state in rank:
        return 0
    
    rank[state] = set()

    if isSolved(state):
        sols.append(state)
        print(f'found sol at depth {depth}')
        return 1

    for neighbor_state in getReachableStates(state):
        rank[state].add(neighbor_state)
        search(neighbor_state, depth + 1, rank, sols, max_search_depth)


if __name__ == '__main__':
    max_search_depth = 20
    num_sols_found = 0



    START_TIME = time.time()

    NUM_WORKERS = cpu_count()

    manager = Manager()
    work_q = Queue()
    rank_d = manager.dict()
    sols = manager.list()
    num_sols_found = manager.list([0])

    work_q.put((INITIAL_STATE, 0))

    procs = [Process(target=search_worker, args=(work_q, rank_d, max_search_depth, sols, num_sols_found, manager)) for _ in range(NUM_WORKERS)]

    print(f'begin search with {NUM_WORKERS} processes')

    for p in procs:
        p.start()
    
    for p in procs:
        p.join()

    END_TIME = time.time()

    rank_vanilla = {}
    sols_vanilla = []
    max_search_depth = max_search_depth

    num_sols_vanilla = search(INITIAL_STATE, 0, rank_vanilla, sols_vanilla, max_search_depth)

    states_searched_multi = set(rank_d.keys())
    states_searched_vanilla = set(rank_vanilla.keys())

    for s in states_searched_vanilla & states_searched_multi:
        n_v = rank_vanilla[s]
        n_m = set(rank_d[s])

        print(f'{len(n_v)}  {len(n_m)}  {len(n_v ^ n_m)}')

    
    print(f'searched {len(rank_d)} unique states to a max depth of {max_search_depth} in {round(END_TIME-START_TIME, 3)} seconds')
    print(f'found {num_sols_found[0]} unique solutions')
    # print(f'shortest sol was {min([rank[s] for s in sols], default=None)}')
