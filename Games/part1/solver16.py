#!/bin/python
# solver16.py : Circular 16 Puzzle solver
# Based on skeleton code by D. Crandall, September 2018
#
import sys
import string

# The final board looks as follows:
final_goal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# The Heuristic function is Manhattan distance with linear conflicts
def Heuristic_function(board):
    Manhattan_distance = 0

    # The co-ordinates of an element in the final state
    goalrow = []    # The row of an element in the final state
    goalcol = []    # The column of an element in the final state

    # For every element in final state, we find the co-ordinates.
    for i in range(16):
        n = (board[i] - 1)//4
        goalrow.append(n)
        m = (board[i] - 1) % 4
        goalcol.append(m)

    # Every element in the current state, we calculate the manhattan distance
    for row in range(4):
        for col in range(4):
            j = row * 4 + col
            Manhattan_row = abs(row - goalrow[j])
            Manhattan_col = abs(col - goalcol[j])
            if Manhattan_row == 3:
                Manhattan_row = 1
            if Manhattan_col == 3:
                Manhattan_col = 1
            Manhattan_distance += Manhattan_row + Manhattan_col
            if goalrow[j] != row:
                continue

            # Applying linear conflicts to the manhattan distance
            # A Linear Conflict is when two elements in the current state are such that
            # their goal positions are on the opposite side of each other
            k = row * 4
            l = k + col
            while k < l:
                if goalrow[k] == row and goalcol[j] < goalcol[k]:
                    Manhattan_distance += 2
                k += 1
    return Manhattan_distance

# shift a specified row left (1) or right (-1)
def shift_row(state, row, dir):
    change_row = state[(row * 4):(row * 4 + 4)]
    return (state[:(row * 4)] + change_row[-dir:] + change_row[:-dir] + state[(row * 4 + 4):],
            ("L" if dir == -1 else "R") + str(row + 1))


# shift a specified col up (1) or down (-1)
def shift_col(state, col, dir):
    change_col = state[col::4]
    s = list(state)
    s[col::4] = change_col[-dir:] + change_col[:-dir]
    return (tuple(s), ("U" if dir == -1 else "D") + str(col + 1))


# pretty-print board state
def print_board(row):
    for j in range(0, 16, 4):
        print('%3d %3d %3d %3d' % (row[j:(j + 4)]))


# return a list of possible successor states
def successors(state):
    return [shift_row(state, i, d) for i in range(0, 4) for d in (1, -1)] + [shift_col(state, i, d) for i in range(0, 4)
                                                                             for d in (1, -1)]


# just reverse the direction of a move name, i.e. U3 -> D3
def reverse_move(state):
    return state.translate(string.maketrans("UDLR", "DURL"))


# check if we've reached the goal
def is_goal(state):
    return sorted(state) == list(state)


# The solver! - using A* using Heuristic function as Manhattan distance with linear conflicts
def solve(initial_board):
    #print(initial_board)
    Heuristic_function(initial_board)
    fringe = [(initial_board, "")]
    while len(fringe) > 0:
        (state, route_so_far) = fringe.pop()
        for (succ, move) in successors(state):
            if is_goal(succ):
                return (route_so_far + " " + move)

            # 7.5 is the mean of the 16 tiles which is used as a factor to increase the efficiency of the Heuristic function
            if 7.5 * Heuristic_function(state) >= 7.5 * Heuristic_function(succ):
                fringe.insert(0, (succ, route_so_far + " " + move))
    return False


# test cases
start_state = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        start_state += [int(i) for i in line.split()]

if len(start_state) != 16:
    print("Error: couldn't parse start state file")

route = solve(tuple(start_state))

print(route)
