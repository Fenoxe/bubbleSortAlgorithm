import time
# import pickle
from collections import Counter
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

    is_empty_bin = [ isEmpty(currState, i) for i in range(numBins) ]
    is_full_bin  = [  isFull(currState, i) for i in range(numBins) ]
    top_ball =     [  getTop(currState, i) for i in range(numBins) ]
    
    for i,j in product(range(numBins), range(numBins)):
        if i == j:
            continue
        if is_empty_bin[i]:
            continue
        if is_full_bin[j]:
            continue
        if is_empty_bin[j] or (top_ball[i] == top_ball[j]):
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


def search(state, from_state, depth, seen, prev, max_depth):

    if depth > max_depth:
        return None, -1

    if state in seen:
        return None, -1
    
    seen[state] = True
    prev[state] = from_state

    if isSolved(state):
        return state,depth

    for neighbor_state in getReachableStates(state):
        s, d = search(neighbor_state, state, depth + 1, seen, prev, max_depth)
        if s:
            return s, d
    
    return None, -1

# gets any possible solution
if __name__ == '__main__':
    seen = {}
    prev = {}
    max_search_depth = 70

    START_TIME = time.time()

    sol,depth = search(INITIAL_STATE, None, 0, seen, prev, max_search_depth)

    END_TIME = time.time()

    # pickle.dump(rank, open('vanilla_rank_dict.p', 'wb'))

    print(f'searched {len(seen)} unique states to a max depth of {max_search_depth} in {round(END_TIME-START_TIME, 3)} seconds')

    if sol:
        print(f'sol was {depth} steps long')
        
        s = sol
        path = []

        while s:
            path.append(s)
            s = prev[s]
        
        path.reverse()

        for i,s in enumerate(path):
            i_str = ' ' * (2 - len(str(i))) + str(i) + ' | '
            s_str = ' '.join([s[4*i : 4*(i+1)] for i in range(len(s) // 4)])

            print(i_str + s_str)