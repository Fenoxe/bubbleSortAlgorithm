import networkx as nx
import time
import pickle
from collections import Counter, deque
import csv
from tqdm import tqdm
from itertools import product
import sys

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

def getReachableStates(state, numBins):

    # create bin heigh and top ball caches
    binHeight,topBall,valid_i,valid_j = getCaches(state, numBins)

    reachable_states = set()

    # search through all possible moves and yield neighbors
    for i in valid_i:

        if state[4 * i] == state[4 * i + 1] == state[4 * i + 2] == state[4 * i + 3]: # bin is completed, don't touch
            continue
            
        for j in valid_j:

            if i == j:
                continue
            
            if binHeight[j] == 0 or topBall[i] == topBall[j]:
                reachable_states.add(moveBall(state, i, j))

    return reachable_states

def isSolved(state, numBins):
    for i in range(numBins):
        if not (state[4 * i] == state[4 * i + 1] == state[4 * i + 2] == state[4 * i + 3]):
            return False
    return True

START_TIME = time.time()
time_total = 0

# define initial game state
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

# INITIAL_STATE = 'prrrllllbwpibgtbeweiypg dig dpg teotow  tdebridwyyy oo  '
INITIAL_STATE = 'prly' + 'oyoy' + 'bwpi' + 'bgtb' + 'ewei' + 'ypgr' + 'digr' + 'dpgl' + 'teot' + 'owll' + 'tdeb' + 'ridw' + '    ' + '    '
# INITIAL_STATE = 'prl ' + '    ' + 'bwpi' + 'bgtb' + 'ewei' + 'ypgr' + 'digr' + 'dpgl' + 'teot' + 'owll' + 'tdeb' + 'ridw' + 'yyy ' + 'oo  '
NUM_BINS = len(INITIAL_STATE) // 4
assert all([(n == 4) for c, n in Counter(INITIAL_STATE).items() if c != ' '])

TIME_SOFT_LIMIT = 3600 / 4
SOLVED_SOFT_LIMIT = 1000

# solve limit check flag
limit_hit = False

# initialize graph and sets
solveGraph = nx.DiGraph()
solveGraph.add_node(INITIAL_STATE)

prevSearched = set()
currSearch = set()
futureSearch = set([INITIAL_STATE])
solvedStates = set()
deadendStates = set()

I = 1

while len(futureSearch) != 0:
    
    print(f'[INFO]  building graph depth={I}')

    currSearch = futureSearch
    futureSearch = set()
    
    s = time.time()

    if TIME_SOFT_LIMIT and (time.time() - START_TIME) > TIME_SOFT_LIMIT:
        print(f'[LIMIT] time soft limit ({TIME_SOFT_LIMIT}) reached')
        limit_hit = True
        if len(solvedStates) > 0:
            break
    
    if SOLVED_SOFT_LIMIT and len(solvedStates) > SOLVED_SOFT_LIMIT:
        print(f'[LIMIT] solved soft limit ({SOLVED_SOFT_LIMIT}) reached')
        limit_hit = True
        break

    for currState in tqdm(currSearch):

        # assert (all([(n == 4) for c, n in Counter(currState).items() if c != ' ']) and len(currState) == NUM_BINS * 4)

        if isSolved(currState, NUM_BINS):
            prevSearched.add(currState)
            
            if currState not in solvedStates:
                tqdm.write(f'[INFO]  found a new solution! ({len(solvedStates) + 1} total found)')
                solvedStates.add(currState)
            
            if limit_hit:
                break

            continue
        
        reachable_states = getReachableStates(currState, NUM_BINS)

        solveGraph.add_edges_from(map(lambda ns: (currState, ns), reachable_states))
        
        futureSearch.update(reachable_states.difference(prevSearched))
        
        prevSearched.add(currState)

    num_nodes = solveGraph.number_of_nodes()
    num_edges = solveGraph.number_of_edges()
    print(f'[INFO]  node count               {num_nodes:,}')
    print(f'[INFO]  edge count               {num_edges:,}')

    time_iter = time.time() - START_TIME - time_total
    time_total = time.time() - START_TIME
    print(f'[INFO]  time iter                {round(time_iter, 3):,} seconds')
    print(f'[INFO]  time elapsed             {round(time_total, 3):,} seconds', end='\n\n\n')
    
    I += 1

num_nodes = solveGraph.number_of_nodes()
num_edges = solveGraph.number_of_edges()

print(f'[POST]  final graph size: {num_nodes} nodes, {num_edges} edges')
print(f'[POST]  found {len(solvedStates)} solutions')
if len(solvedStates) == 0:
    print('lol rip, exiting...')
    exit()
print(f'[POST]  finding best solution...')

shortestPath = None
shortestPath_l = float('inf')

for finalState in tqdm(solvedStates):
    shortestPathCurr = nx.algorithms.bidirectional_shortest_path(solveGraph, INITIAL_STATE, finalState)

    if (l := len(shortestPathCurr)) < shortestPath_l:
        shortestPath = shortestPathCurr
        shortestPath_l = l

print(f'[POST]  found optimal solution')
print(f'[POST]  # of moves = {shortestPath_l}', end='\n\n')

for state in shortestPath:
    print(state, end='\n\n')

END_TIME = time.time()

print(f'[POST]  algo finished in {round(END_TIME - START_TIME, 2)} seconds')

with open('moves.pickle', 'wb') as buf:
    pickle.dump(shortestPath, buf, protocol=pickle.HIGHEST_PROTOCOL)

print(f'[POST]  saved to pickled file')
