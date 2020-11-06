import networkx as nx
import time
import pickle
from collections import Counter
import csv
from tqdm import tqdm
from itertools import product
import sys

# recursive getsizeof
def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size

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

if __name__ == '__main__':

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
    INITIAL_STATE = 'prly' + 'oyoy' + 'bwpi' + 'bgtb' + 'ewei' + 'ypgr' + 'digr' + 'dpgl' + 'teot' + 'owll' + 'tdeb' + 'ridw' + '    ' + '    '
    NUM_BINS = len(INITIAL_STATE) // 4
    assert all([(n == 4) for c, n in Counter(INITIAL_STATE).items() if c != ' '])

    # solve limits and reporting config
    NODE_SOFT_LIMIT = 250000 * NUM_BINS
    TIME_SOFT_LIMIT = 60 * 60

    # solve limit check flag
    limit_hit = False

    # initialize graph and sets
    solveGraph = nx.DiGraph()
    addNode(solveGraph, INITIAL_STATE)

    prevSearched = set()
    currSearch = set()
    futureSearch = set([INITIAL_STATE])
    solvedStates = set()

    I = 0

    while len(futureSearch) != 0:
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

        for currState in tqdm(currSearch):

            # assert (all([(n == 4) for c, n in Counter(currState).items() if c != ' ']) and len(currState) == NUM_BINS * 4)

            if limit_hit and len(solvedStates) > 0:
                break

            if isSolved(currState, NUM_BINS):
                prevSearched.add(currState)
                solvedStates.add(currState)

                print(f'[INFO]  found a solution! ({len(solvedStates)} total found)')

                if limit_hit:
                    break

                continue

            for reachableState in getReachableStates(currState, NUM_BINS):
                addEdge(solveGraph, currState, reachableState)
                
                if reachableState not in prevSearched:
                    futureSearch.add(reachableState)

            prevSearched.add(currState)

        # this shit slows down the algo hella
        print(f'[INFO]  node count               {solveGraph.number_of_nodes():,}')
        print(f'[INFO]  edge count               {solveGraph.number_of_edges():,}')
        print(f'[INFO]  sizeof graph (bytes)     {get_size(list(solveGraph.edges.items())) + get_size(list(solveGraph.nodes.items())):,}')
        print(f'[INFO]  sizeof sets (bytes)      {sum([get_size(s) for s in (prevSearched,currSearch,futureSearch,solvedStates)]):,}')
        print(f'[INFO]  time elapsed             {int(round(time.time()-START_TIME)):,} seconds', end='\n\n\n')

        I += 1

    print(f'[POST]  finished building game state tree')
    print(f'[POST]  final graph size = {len(solveGraph)}')
    print(f'[POST]  finding best solution...')

    if limit_hit:
        shortestPath = None
        shortestPath_l = float('inf')

        for finalState in tqdm(solvedStates):
            shortestPathCurr = nx.algorithms.bidirectional_shortest_path(solveGraph, INITIAL_STATE, finalState)

            if (l := len(shortestPathCurr)) < shortestPath_l:
                shortestPath = shortestPathCurr
                shortestPath_l = l
        
    else:
        for _, finalState in tqdm(nx.algorithms.bfs_edges(solveGraph, INITIAL_STATE)):
            if isSolved(finalState, NUM_BINS):
                break
        
        shortestPath = nx.algorithms.bidirectional_shortest_path(solveGraph, INITIAL_STATE, finalState)
        shortestPath_l = len(shortestPath)

    print('[POST]  found optimal solution')
    print(f'[POST]  # of moves = {shortestPath_l}')

    for d, state in enumerate(shortestPath):
        print(f'{d:>2}: {state}', end='\n\n')

    END_TIME = time.time()

    print(f'[POST]  algo finished in {END_TIME - START_TIME} seconds')

    filename = 'moves.pickle'

    with open(filename, 'wb') as buf:
        pickle.dump(shortestPath, buf, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'[POST]  saved optimal state path to {filename}')
