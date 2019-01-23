#!/usr/bin/env python3
import sys
import copy

# generate the 2*N successors for a given state
def successor(current_state, N, V):
    list_successors = []
    head_cols = [-1]*N 
    # to get the top location of a piece in each column
    for r in range(N+3) :
        for c in range(N) :
            if r !=0 and current_state[r][c] == 0 and current_state[r-1][c] != 0 :
                head_cols[c] = r-1  #can skip this column from next loop
            elif r == N+2 and current_state[r][c] !=0  :
                head_cols[c] = r       
            elif r == 0 and current_state[r][c] != 0 :
                head_cols[c] = 0 
    # rotate operations
    for c in range(N) :
        successor_state = copy.deepcopy(current_state)
        top = successor_state[0][c]
        last = head_cols[c]
        for r in range(last) :   
            successor_state[r][c] = successor_state[r+1][c]
        successor_state[last][c]=top    
        list_successors.append(successor_state)
    # drop operations
    count = 0
    for r in range(N+3):
        for c in range(N):
            if current_state[r][c] == 1 :
                count+=1
    # if count exceeds max pieces don't perform drop operation
    if count <= ((N*(N+3))/2) :
        for c in range(N) :
            top = head_cols[c]
            if top < N+2 :
                successor_state = copy.deepcopy(current_state)
                successor_state[top+1][c] = V # if my piece/ my turn  V is 1 else -1 if opponent's turn
                list_successors.append(successor_state)
    return list_successors

# get the cost of a state using heuristic
# for my heuristic, i am getting count of the total number of my pieces in only the change affected for a row and column and if it is in any of the 2 diagonals
# i am giving high cost if my move results in a win
# also giving high negative cost to opponents win
def get_cost(new_state,dropped_c,r_effected) :
    affected_cost=0    
    for r in range(3,N+3) :
        r_sum = 0
        or_sum = 0
        if r == r_effected :
            for c in range(N) :
                if new_state[r][c] == 1 :
                    affected_cost += 1    
        for c in range(N) :
            if new_state[r][c] == 1 :
                r_sum += 1
            elif new_state[r][c] == -1 :
                or_sum += 1
        if r_sum == N :
            affected_cost += 1000*N
            return affected_cost
        elif or_sum == N :
            affected_cost += -1000*N
            return affected_cost
    for c in range(N) :
        c_sum = 0
        oc_sum = 0
        d1_sum = 0
        od1_sum = 0
        d2_sum = 0
        od2_sum = 0
        if c == dropped_c :
            for r in range(3,N+3) :
                if new_state[r][c] == 1 :
                    affected_cost += 1       
                if new_state[r][c] == 1 and r == r_effected and c == (r-3) : # diagonal
                    affected_cost += 1
                    d1_sum += 1
                if new_state[r][c] == 1 and r == r_effected and c == (N-r+2) : # diagonal
                    affected_cost += 1
                    d2_sum += 1
                if new_state[r][c] == -1 and r == r_effected and c == (r-3) : # diagonal
                    od1_sum += 1
                if new_state[r][c] == -1 and r == r_effected and c == (N-r+2) : # diagonal 
                    od2_sum += 1
        for r in range(3,N+3) :
            if new_state[r][c] == 1 :
                c_sum += 1
            elif new_state[r][c] == -1 :
                oc_sum += 1
        if c_sum == N or d1_sum == N or d2_sum == N :
            affected_cost += 1000*N
            return affected_cost
        elif oc_sum == N or od1_sum == N or od2_sum == N :
            affected_cost += -1000*N
            return affected_cost
    return affected_cost

#print the board in desired format     
def print_board(state,move,my_symbol) :
    state = state[::-1]
    print(move,end='')
    print(" ",end='')
    if my_symbol == 'x' :
        opp_symbol = 'o'
    else :
        opp_symbol = 'x'
    for r in range(N+3) :
        for c in range(N) :
            if state[r][c] == 1 :
                print(my_symbol,end='')
            elif state[r][c] == 0 :
                print('.',end='')
            else :
                print(opp_symbol,end='')

# check if a state is goal
def is_goal(current_state,N):
    sum_c = [0]*N
    sum_d1 = 0
    sum_d2 = 0
    for r in range(3,N+3) :
        sum_r = 0
        for c in range(N) :
            sum_r += current_state[r][c]      
            sum_c[c] += current_state[r][c] 
            if r+c == N+2 :                     
                sum_d1 += current_state[r][c]
            if r-3 == c :
                sum_d2 += current_state[c][c]
            if r == N+2 and sum_c[c] == N :
                return True 
        if sum_r == N :
            return True
    if sum_d1 == N or sum_d2 == N :
            return True
    return False

#implementation of alpha-beta pruning
def Alpha_Beta(s, N, D, NBN, PBN, my_piece):
    list_suc = []
    list_next_state = []
    list_next_state_temp = []
    list_next_state_temp_c = []
    list_succ = []
    list_cost_suc = []
    vc = 0
    vr = 0
    for suc in successor(s, N, 1) : 
        list_succ.append(suc)
        list_suc.append(min_val(suc,N,D,NBN,PBN))
        if vc > N-1 :
            vr += 1
        else :
            vr = 2
        list_cost_suc.append(get_cost(suc, vc, vr)) 
        vc += 1
        
        # to keep printing best state explored so far - if timer expires
        list_next_state_temp = list_succ[list_cost_suc.index(max(list_cost_suc))]
        list_next_state_temp_c = list_cost_suc.index(max(list_cost_suc))
        next_state_temp = list_next_state_temp
        move_t = int(list_next_state_temp_c)
        if move_t < N :
            move_t = (move_t+1)*-1
        else :
            move_t = move_t+1-N
        print_board(next_state_temp,move_t,my_piece)
        print()
                
    list_next_state.append(list_succ[list_cost_suc.index(max(list_cost_suc))])
    list_next_state.append(list_cost_suc.index(max(list_cost_suc)))
    return list_next_state
           
def max_val(s, N, D, A, B):
    if (is_goal(s, N) and s != None) :
        return s
    elif D > -1 :
        list_suc = []
        list_cost_suc = []
        vc = 0
        vr = 0
        D -= 1
        for suc in successor(s, N, 1) : 
            list_suc.append(min_val(suc,N,D,A,B))
            if vc > N-1 :
                vr += 1
            else :
                vr = 2
            list_cost_suc.append(get_cost(suc, vc, vr)) 
            vc += 1
            
            A = max(A,max(list_cost_suc))
            if A >= B :
                return A
    return A

def min_val(s, N, D, A, B):
    if (is_goal(s, N) and s != None) :
        return s
    elif D > -1 :
        list_suc = []
        list_cost_suc = []
        vc = 0
        vr = 0
        D -= 1
        for suc in successor(s, N, -1) : 
            list_suc.append(max_val(suc,N,D,A,B))
            if vc > N-1 :
                vr += 1
            else :
                vr = 2
            list_cost_suc.append(get_cost(suc, vc, vr)) 
            vc += 1
            B = min(B,min(list_cost_suc))
            if A >= B :
                return B
    return B

       
N = int(sys.argv[1])        # size of board - (N+3)*N
my_piece = (sys.argv[2])
board = (sys.argv[3])
board = board[::-1]     #reverse the board

PBN = 10000000000   # some big number to represent infinity
NBN = -1*PBN     
current_state = [[-1]*N for i in range(N+3)]
# encoding the board to 1(my piece), 0(empty), -1(opponent's piece)
for r in range(N+3) :
    for c in range(N) :
        if board[c+(r*N)] == my_piece :
            current_state[r][c] = 1
        elif board[c+(r*N)] == '.' :
            current_state[r][c] = 0
# transpose                
for r in range(N+3) :
    for c in range(int(N/2)) :
        temp = current_state[r][c]
        current_state[r][c] = current_state[r][N-1-c]
        current_state[r][N-1-c] = temp

# depth to cut-off
D = 4 # must be dynamic based on cut-off time and size of board
next_state = Alpha_Beta(current_state, N, D, NBN, PBN, my_piece)
move = next_state[-1]
del next_state[-1]
next_state = next_state[0]
if move < N :
    move = (move+1)*-1
else :
    move = move+1-N
print_board(next_state,move,my_piece) #print the final board after making my move
print()
