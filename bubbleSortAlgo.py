import networkx as nx
import time
import pickle
from collections import Counter
import csv
from tqdm import tqdm

class LogEntry:

    def __init__(self):
        self.checks =                  0
        self.SA_pre =                  0
        self.SA_calc_reachable =       0
        self.SA_add_nodes =            0
        self.SA_set_logic =            0
        self.num_searches =            0
        self.num_solved   =            0
        self.num_new_states =          0

    def getDataRow(self):
        return  [   self.checks,
                    self.SA_pre,
                    self.SA_calc_reachable,
                    self.SA_add_nodes,
                    self.SA_set_logic,
                    self.num_searches,
                    self.num_solved,
                    self.num_new_states,
                ]

def getTop(state, i):
    ball = ' '
    for o in range(3, -1, -1):
        if (ball := state[4 * i + o]) != ' ':
            break
    return ball

def setTop(state, i, ball):
    for o in range(3, -1, -1):
        pos = 4 * i + o
        belowPos = pos - 1

        if (state[pos] == ' ' and state[belowPos] != ' ') or (o == 0):
            if pos == len(state) - 1:
                newState = state[:pos] + ball
            else:
                newState = state[:pos] + ball + state[pos + 1:]
            break
    
    return newState

def remTop(state, i):
    for o in range(3, -1, -1):
        pos = 4 * i + o

        if state[pos] != ' ':
            if pos == len(state) - 1:
                newState = state[:pos] + ' '
            else:
                newState = state[:pos] + ' ' + state[pos + 1:]
            break
    
    return newState

def isEmpty(state, i):
    return all([(state[4 * i + o] == ' ') for o in range(4)])

def isFull(state, i):
    return all([(state[4 * i + o] != ' ') for o in range(4)])

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
    reachableStates = []
    
    for i in range(numBins):
        for j in range(numBins):
            if i == j:
                continue
            if isEmpty(currState, i):
                continue
            if isFull(currState, j):
                continue
            if isEmpty(currState, j) or (getTop(currState, i) == getTop(currState, j)):
                newState = moveBall(currState, i, j)
                reachableStates.append(newState)

    return set(reachableStates)

def isSolved(state, numBins):
    for i in range(numBins):
        t = list(state[4*i:4*i+4])
        if len(set(t)) > 1:
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
    alert_vals = [25, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 500000, 1000000]
    curr_pt = 0

    # solve limit check -do not change-
    limit_hit = False

    # initialize graph and sets
    solveGraph = nx.DiGraph()
    prevSearched = set()
    currSearch = set()
    futureSearch = set([INITIAL_STATE])
    solvedStates = set()

    # initialize logging
    log = [LogEntry()]
    I = 0

    while len(futureSearch) != 0:
        print(f'[INFO]  starting search for depth={I+1}')

        currSearch = futureSearch
        futureSearch = set()
        
        s = time.time()

        if curr_pt < len(alert_vals) and (l := len(solveGraph)) > (a_l := alert_vals[curr_pt]):
            print(f'[LIMIT] graph exceeded {a_l} nodes')
            curr_pt += 1
        
        if NODE_SOFT_LIMIT and l > NODE_SOFT_LIMIT:
            print(f'[LIMIT] node soft limit ({NODE_SOFT_LIMIT}) reached')
            limit_hit = True
            if len(solvedStates) > 0:
                break

        if TIME_SOFT_LIMIT and (time.time() - START_TIME) > TIME_SOFT_LIMIT:
            print(f'[LIMIT] time soft limit ({TIME_SOFT_LIMIT}) reached')
            limit_hit = True
            if len(solvedStates) > 0:
                break
        
        e = time.time()
        log[I].checks = e - s
        
        log[I].num_searches = len(currSearch)

        for currState in tqdm(currSearch):
            s = time.time()

            if limit_hit and len(solvedStates) > 0:
                break

            if isSolved(currState, NUM_BINS):
                prevSearched.add(currState)
                solvedStates.add(currState)

                print(f'[INFO]  found a solution! ({len(solvedStates)} total found so far)')
                log[I].num_solved += 1

                if limit_hit:
                    break

                continue

            e = time.time()
            log[I].SA_pre += e - s

            s = time.time()

            reachableStates = getReachableStates(currState, NUM_BINS)

            e = time.time()
            log[I].SA_calc_reachable += e - s

            s = time.time()

            for reachableState in reachableStates:
                addEdge(solveGraph, currState, reachableState)

            e = time.time()
            log[I].SA_add_nodes += e - s
            s = time.time()

            prevSearched.add(currState)

            newStates = reachableStates.difference(prevSearched)

            futureSearch.update(newStates)

            e = time.time()

            log[I].SA_set_logic += e - s
            log[I].num_new_states += len(newStates)

        print(f'[INFO]  time elapsed = {int(round(time.time()-START_TIME))} seconds')
        print()

        I += 1
        log.append(LogEntry())

    print()
    print(f'[POST]  final graph size = {len(solveGraph)}')
    print(f'[POST]  finding best solution...')

    if limit_hit:
        shortestPath = None
        shortestPath_l = float('inf')

        print()
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

    print(f'[POST]  found best solution')
    print(f'[POST]  least # of moves = {shortestPath_l}')
    print()

    for state in shortestPath:
        print(state)
        print()

    END_TIME = time.time()

    print(f'[POST]  algo finished in {END_TIME - START_TIME} seconds')

    with open('moves.pickle', 'wb') as buf:
        pickle.dump(shortestPath, buf, protocol=pickle.HIGHEST_PROTOCOL)

    with open('iter_data.csv', mode='w') as iter_data:
        iter_data_writer = csv.writer(  iter_data,
                                        delimiter=',',
                                        quotechar='"',
                                        quoting=csv.QUOTE_NONNUMERIC
                                    )

        iter_data_writer.writerow([ 'Duration of Limit Checking',
                                    'Total Duration of Search Prechecks',
                                    'Total Duration of Calculating Reachable States',
                                    'Total Duration of Adding Nodes',
                                    'Total Duration of Set Logic',
                                    'Number of States to Search',
                                    'Number of New Solved States Found',
                                    'Number of New States Found'
                                    ])
        for logEntry in log:
            iter_data_writer.writerow(logEntry.getDataRow())

    print(f'[POST]  saved moves and algo data to csv files')
