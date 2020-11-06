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

def addEdge(G, currState, nextState):
    G.add_edge(currState, nextState)

def addNode(G, state):
    G.add_node(state)

def remEdge(G, currState, nextState):
    G.remove_edge(currState, nextState)

def remNode(G, state):
    G.remove_node(state)

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
    print(state)
    return True


if __name__ == '__main__':
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

    INITIAL_STATE = 'prrrllllbwpibgtbeweiypg dig dpg teotow  tdebridwyyy oo  '
    INITIAL_STATE = 'prly' + 'oyoy' + 'bwpi' + 'bgtb' + 'ewei' + 'ypgr' + 'digr' + 'dpgl' + 'teot' + 'owll' + 'tdeb' + 'ridw' + '    ' + '    '
    INITIAL_STATE = 'prl ' + '    ' + 'bwpi' + 'bgtb' + 'ewei' + 'ypgr' + 'digr' + 'dpgl' + 'teot' + 'owll' + 'tdeb' + 'ridw' + 'yyy ' + 'oo  '
    NUM_BINS = len(INITIAL_STATE) // 4
    assert all([(n == 4) for c, n in Counter(INITIAL_STATE).items() if c != ' '])

    # solve limits and reporting config
    NODE_SOFT_LIMIT = 500000 * NUM_BINS
    TIME_SOFT_LIMIT = 60 * 60
    SOLVED_SOFT_LIMIT = 100
    GRAPH_INFO_LOGGING = False

    # solve limit check flag
    limit_hit = False

    # initialize graph and sets
    solveGraph = nx.DiGraph()
    addNode(solveGraph, INITIAL_STATE)

    prevSearched = set()
    currSearch = set()
    futureSearch = set([INITIAL_STATE])
    solvedStates = set()
    deadendStates = set()

    I = 1

    # temporary limit on depth for testing
    I_HARD_LIMIT = None

    if GRAPH_INFO_LOGGING:
        # temporary data logging for pruning purposes
        pruning_data_buf = open('pruning_data.csv', 'a', newline='')
        pruning_data_writer = csv.writer(
            pruning_data_buf,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_NONE,
        )
        pruning_data_writer.writerow([
            'depth',
            'num_nodes',
            'num_edges',
            'mem_graph',
            'mem_sets',
            'time_iter',
            'time_total',
        ])

    while len(futureSearch) != 0:
        if I_HARD_LIMIT and I > I_HARD_LIMIT:
            if GRAPH_INFO_LOGGING:
                # write new line in data
                pruning_data_writer.writerow('')
            exit()
        
        print(f'[INFO]  building graph depth={I}')

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
        
        if SOLVED_SOFT_LIMIT and len(solvedStates) > SOLVED_SOFT_LIMIT:
            print(f'[LIMIT] solved soft limit ({SOLVED_SOFT_LIMIT}) reached')
            limit_hit = True
            break

        for currState in tqdm(currSearch):

            assert (all([(n == 4) for c, n in Counter(currState).items() if c != ' ']) and len(currState) == NUM_BINS * 4)

            if limit_hit and len(solvedStates) > 0:
                break

            if isSolved(currState, NUM_BINS):
                prevSearched.add(currState)
                
                if currState not in solvedStates:
                    tqdm.write(f'[INFO]  found a new solution! ({len(solvedStates) + 1} total found)')
                    solvedStates.add(currState)
                
                if limit_hit:
                    break

                continue
            
            num_reachable_states = 0
            for reachableState in getReachableStates(currState, NUM_BINS):
                num_reachable_states += 1

                if reachableState not in deadendStates:
                    addEdge(solveGraph, currState, reachableState)
                
                    if reachableState not in prevSearched:
                        futureSearch.add(reachableState)

            if num_reachable_states == 0:
                deadendStates.add(currState)
                if currState in solveGraph:
                    remNode(solveGraph, currState)

            prevSearched.add(currState)

        if GRAPH_INFO_LOGGING:
            # this shit slows down the algo hella
            num_nodes = solveGraph.number_of_nodes()
            num_edges = solveGraph.number_of_edges()
            mem_graph = get_size(list(solveGraph.edges.items())) + get_size(list(solveGraph.nodes.items()))
            mem_sets  = sum([get_size(s) for s in (prevSearched,currSearch,futureSearch,solvedStates,deadendStates)])
            print(f'[INFO]  node count               {num_nodes:,}')
            print(f'[INFO]  edge count               {num_edges:,}')
            print(f'[INFO]  sizeof graph (bytes)     {mem_graph:,}')
            print(f'[INFO]  sizeof sets (bytes)      {mem_sets:,}')

        time_iter = time.time() - START_TIME - time_total
        time_total = time.time() - START_TIME
        print(f'[INFO]  time iter                {round(time_iter, 3):,} seconds')
        print(f'[INFO]  time elapsed             {round(time_total, 3):,} seconds', end='\n\n\n')

        if GRAPH_INFO_LOGGING:
            pruning_data_writer.writerow([
                I,
                num_nodes,
                num_edges,
                mem_graph,
                mem_sets,
                time_iter,
                time_total,
            ])
        
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

    '''
    num_paths = 0
    max_path_len = 0
    for path in tqdm(nx.algorithms.all_simple_paths(solveGraph, INITIAL_STATE, finalState)):
        num_paths += 1
        max_path_len = max(len(path), max_path_len)

    print(f'[FUN]  number of ways to reach optimal solution:      {num_paths}')
    print(f'[FUN]  length of longest path to optimal solution:    {max_path_len}')
    '''

    for state in shortestPath:
        print(state, end='\n\n')

    END_TIME = time.time()

    print(f'[POST]  algo finished in {round(END_TIME - START_TIME, 2)} seconds')

    with open('moves.pickle', 'wb') as buf:
        pickle.dump(shortestPath, buf, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'[POST]  saved to pickled file')
