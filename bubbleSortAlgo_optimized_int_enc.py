import networkx as nx
import time
import pickle
from collections import Counter
import csv
from tqdm import tqdm
from itertools import product

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
        return state[:4 * i] + ball + state[4 * i + 1:]
    if state[4 * i + 1] == ' ':
        return state[:4 * i + 1] + ball + state[4 * i + 2:]
    if state[4 * i + 2] == ' ':
        return state[:4 * i + 2] + ball + state[4 * i + 3:]
    return state[:4 * i + 3] + ball + state[4 * i + 4:]

def remTop(state, i):
    if state[4 * i + 3] != ' ':
        return state[:4 * i + 3] + ' ' + state[4 * i + 4:]
    if state[4 * i + 2] != ' ':
        return state[:4 * i + 2] + ' ' + state[4 * i + 3:]
    if state[4 * i + 1] != ' ':
        return state[:4 * i + 1] + ' ' + state[4 * i + 2:]
    return state[:4 * i] + ' ' + state[4 * i + 1:]

def isEmpty(state, i):
    return state[4 * i] == ' '

def isFull(state, i):
    return state[4 * i + 3] != ' '

def moveBall(state, i, j):
    ball = getTop(state, i)
    newState = setTop(state, j, ball)
    newState = remTop(newState, i)
    return newState

def addEdge(G, currState, nextState):
    G.add_edge(currState, nextState)

def addNode(G, state):
    G.add_node(state)

def getReachableStates(currState, numBins):
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

def isSolved(state, numBins):
    for i in range(numBins):
        bin_t = state[4 * i:4 * i + 4]
        if bin_t != 4 * bin_t[0]:
            return False
    return True


# base 13 encoding of state string

encoding_pairs = (
    (' ', 0),
    ('p', 1),
    ('r', 2),
    ('l', 3),
    ('y', 4),
    ('o', 5),
    ('b', 6),
    ('w', 7),
    ('i', 8),
    ('t', 9),
    ('g', 10),
    ('e', 11),
    ('d', 12),
)

char_to_digit = {c: i for c,i in encoding_pairs}
digit_to_char = {i: c for c,i in encoding_pairs}
BASE = len(encoding_pairs)

def str_to_int(s):
    assert type(s) == str
    i = 0
    p = 1
    for c in s:
        i += p * char_to_digit[c]
        p *= BASE
    return i

def int_to_str(i):
    assert type(i) == int

    s = ''
    p = BASE ** (56 - 1)
    for _ in range(56):
        d = i // p
        s = digit_to_char[d] + s
        i = i % p
        p = p // BASE
    return s


START_TIME = time.time()

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
INITIAL_STATE_s = 'prly' + 'oyoy' + 'bwpi' + 'bgtb' + 'ewei' + 'ypgr' + 'digr' + 'dpgl' + 'teot' + 'owll' + 'tdeb' + 'ridw' + '    ' + '    '
NUM_BINS = len(INITIAL_STATE_s) // 4
assert all([(n == 4) for c, n in Counter(INITIAL_STATE_s).items() if c != ' '])
INITIAL_STATE_i = str_to_int(INITIAL_STATE_s)

# solve limits and reporting config
NODE_SOFT_LIMIT = 250000 * NUM_BINS
TIME_SOFT_LIMIT = 60 * 60

# solve limit check flag
limit_hit = False

# initialize graph and sets
solveGraph = nx.DiGraph() # contains strs encoded as ints
addNode(solveGraph, INITIAL_STATE_i)

prevSearched = set() # contains strs encoded as ints
currSearch = set() # contains strs encoded as ints
futureSearch = set([INITIAL_STATE_i]) # contains strs encoded as ints
solvedStates = set() # contains strs encoded as ints

I = 0

while len(futureSearch) > 0:
    print(f'[INFO]  starting search for depth={I+1}')

    currSearch = futureSearch
    futureSearch = set()
    
    s = time.time()

    if NODE_SOFT_LIMIT and len(solveGraph) > NODE_SOFT_LIMIT:
        print(f'[LIMIT] node soft limit ({NODE_SOFT_LIMIT}) reached')
        limit_hit = True
        if len(solvedStates) > 0:
            break

    if TIME_SOFT_LIMIT and (time.time() - START_TIME) > TIME_SOFT_LIMIT:
        print(f'[LIMIT] time soft limit ({TIME_SOFT_LIMIT}) reached')
        limit_hit = True
        if len(solvedStates) > 0:
            break

    for currState_i in tqdm(currSearch):

        assert (all([(n == 4) for c, n in Counter(int_to_str(currState_i)).items() if c != ' ']) and len(int_to_str(currState_i)) == NUM_BINS * 4)

        if limit_hit and len(solvedStates) > 0:
            break

        if isSolved(int_to_str(currState_i), NUM_BINS):
            prevSearched.add(currState_i)
            solvedStates.add(currState_i)

            print(f'[INFO]  found a solution! ({len(solvedStates)} total found)')

            if limit_hit:
                break

            continue

        for reachableState_s in getReachableStates(int_to_str(currState_i), NUM_BINS):
            reachableState_i = str_to_int(reachableState_s)
            addEdge(solveGraph, currState_i, reachableState_i)
            
            if reachableState_i not in prevSearched:
                futureSearch.add(reachableState_i)

        prevSearched.add(currState_i)

    print(f'[INFO]  time elapsed = {int(round(time.time()-START_TIME))} seconds')
    print()

    I += 1

print()
print(f'[POST]  final graph size = {len(solveGraph)}')
print(f'[POST]  finding best solution...')

if limit_hit:
    shortestPath = None
    shortestPath_l = float('inf')

    for finalState_i in tqdm(solvedStates):
        shortestPathCurr = nx.algorithms.bidirectional_shortest_path(solveGraph, INITIAL_STATE_i, finalState_i)

        if (l := len(shortestPathCurr)) < shortestPath_l:
            shortestPath = shortestPathCurr
            shortestPath_l = l
    
else:
    for _, finalState_i in tqdm(nx.algorithms.bfs_edges(solveGraph, INITIAL_STATE_i)):
        if isSolved(int_to_str(finalState_i), NUM_BINS):
            break
    
    shortestPath_i = nx.algorithms.bidirectional_shortest_path(solveGraph, INITIAL_STATE_i, finalState_i)
    shortestPath_l = len(shortestPath_i)

print(f'[POST]  found optimal solution')
print(f'[POST]  # of moves = {shortestPath_l}')
print()

for d, state_i in enumerate(shortestPath_i):
    print(f'{d}: {int_to_str(state_i)}', end='\n\n')

END_TIME = time.time()

print(f'[POST]  algo finished in {END_TIME - START_TIME} seconds')

with open('moves.pickle', 'wb') as buf:
    pickle.dump(shortestPath_i, buf, protocol=pickle.HIGHEST_PROTOCOL)

print(f'[POST]  saved to pickled file')
