# Make sure to provide the path of the road-segments text file
# This is the routing program for different algorithms using different cost functions
import sys
from queue import PriorityQueue
from heapq import heappush, heappop

def successors_bfs(city):
    next_cities = []
    for item in road_segments:
        if item[1] == city[1] or item[1] == city[1]:
            next_cities.append(item)
    return next_cities
def successors_bfs_distance(city):
    next_cities = []
    next_cities_dist = []
    city_min_dist = 0
    for item in road_segments:
        if item[0] == city[1] or item[1] == city[1]:
            next_cities.append(item)
            next_cities_dist.append(item[2])
    next_cities_min = min(next_cities_dist)
    for item in next_cities:
        if next_cities_min == item[2]:
            if item[0] == city:
                city_min_dist = item[1]
            if item[1] == city:
                city_min_dist = item[0]
    return city_min_dist

def successors_bfs_time(city):
    next_cities = []
    next_cities_time = []
    city_min_time = 0
    for item in road_segments:
        if item[0] == city[1] or item[1] == city[1]:
            next_cities.append(item)
            time = float(item[2]) / float(item[3])
            next_cities_time.append(time)
    next_cities_min = min(next_cities_time)
    i = next_cities_time.index(next_cities_min)
    min_time_tuple = next_cities[i]
    if min_time_tuple[0] == city:
        city_min_time = min_time_tuple[1]
    if min_time_tuple[1] == city:
        city_min_time = min_time_tuple[0]
    return city_min_time

def successors_uniform(city):
    city_tuple = city[1]
    next_cities = []
    for item in road_segments:
        if item[0] == city_tuple[1] or item[1] == city_tuple[1]:
            next_cities.append(item)
    return next_cities

def successors_astar_distance(city, pre_city):
    city_tuple = city[1]
    next_cities = []
    next_cities_dist = []
    city_min_dist = 0
    if city_tuple[0] == pre_city:
        temp = city_tuple[1]
    else:
        temp = city_tuple[0]
    for item in road_segments:
        if ((item[0] == pre_city and item[1] != temp) or (item[1] == pre_city and item[0] != temp)):
            next_cities.append(item)
            next_cities_dist.append(item[2])
    next_cities_min = min(next_cities_dist)
    for item in next_cities:
        if next_cities_min == item[2]:
            if item[0] == city:
                city_min_dist = item[1]
            if item[1] == city:
                city_min_dist = item[0]
    return city_min_dist

def successors_astar_time(city, pre_city):
    city_tuple = city[1]
    next_cities = []
    next_cities_time = []
    city_min_time = 0
    if city_tuple[0] == pre_city:
        temp = city_tuple[1]
    else:
        temp = city_tuple[0]
    for item in road_segments:
        if ((item[0] == pre_city and item[1] != temp) or (item[1] == pre_city and item[0] != temp)):
            next_cities.append(item)
            time = float(item[2])/float(item[3])
            next_cities_time.append(time)

    next_cities_min = min(next_cities_time)
    i = next_cities_time.index(next_cities_min)
    min_time_tuple = next_cities[i]
    if min_time_tuple[0] == city:
        city_min_time = min_time_tuple[1]
    if min_time_tuple[1] == city:
        city_min_time = min_time_tuple[0]
    return city_min_time

def successors_astar_segments(city, pre_city):
    city_tuple = city[1]
    next_cities = []
    if city_tuple[0] == pre_city:
        temp = city_tuple[1]
    else:
        temp = city_tuple[0]
    for item in road_segments:
        if ((item[0] == pre_city and item[1] != temp) or (item[1] == pre_city and item[0] != temp)):
            next_cities.append(item)
    return

def successors_dfs(city, pre_city):
    city_tuple = city[1]
    next_cities = []
    if city_tuple[0] == pre_city:
        temp = city_tuple[1]
    else:
        temp = city_tuple[0]
    for item in road_segments:
        if ((item[0] == pre_city and item[1] != temp) or (item[1] == pre_city and item[0] != temp)):
            next_cities.append(item)
    return next_cities


def is_goal(city):
    return (city[0] == destination_city or city[1] == destination_city)


def solve_ids(initial_city, cost):
    k = 5
    while 1:
        if solve_idfs(initial_city, k, cost):
            print("Found using IDS")
            return True
        else:
            k += 5

def solve_idfs(initial_city, k, cost):
    visited_cities = []
    counter = -1
    c = 0
    city_tuple = []
    for item in road_segments:
        if item[0] == initial_city or item[1] == initial_city:
            city_tuple.append(item)

    fringe = []
    visited_cities.append(initial_city)
    for item in city_tuple:
        if is_goal(item):
            print("Found using IDS")
            return True
        heappush(fringe, (counter, item))
    prev_city = initial_city

    while len(fringe) > 0:

        state = heappop(fringe)
        heappush(fringe, (0, state[1]))
        temp = state[1]

        if temp[0] == prev_city:
            pre_city = temp[1]
        else:
            pre_city = temp[0]
        counter -= 1

        if pre_city not in visited_cities:
            if cost == "distance":
                for succ in successors_astar_distance(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
            if cost == "time":
                for succ in successors_astar_time(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
            if cost == "segments":
                for succ in successors_dfs(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
                        c += 1
            visited_cities.append(pre_city)
            prev_city = pre_city
    if k > (-1 * counter):
        return False


def solve_dfs(initial_city, cost):
    visited_cities = []
    counter = -1
    c = 0
    city_tuple = []
    for item in road_segments:
        if item[0] == initial_city or item[1] == initial_city:
            city_tuple.append(item)

    fringe = []
    visited_cities.append(initial_city)
    for item in city_tuple:
        if is_goal(item):
            print("Found using DFS")
            return True
        heappush(fringe, (counter, item))
    prev_city = initial_city

    while len(fringe) > 0:

        state = heappop(fringe)

        heappush(fringe, (0, state[1]))
        temp = state[1]

        if temp[0] == prev_city:
            pre_city = temp[1]
        else:
            pre_city = temp[0]
        counter -= 1
        
        if pre_city not in visited_cities:
            if cost == "distance":
                for succ in successors_astar_distance(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
            if cost == "time":
                for succ in successors_astar_time(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
            if cost == "segments":
                for succ in successors_dfs(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
                        c += 1
            visited_cities.append(pre_city)
            prev_city = pre_city
    return False


def solve_astar(initial_city, cost):
    visited_cities = []
    counter = -1
    c = 0
    city_tuple = []

    for item in road_segments:
        if item[0] == initial_city or item[1] == initial_city:
            city_tuple.append(item)

    fringe = []
    visited_cities.append(initial_city)

    for item in city_tuple:
        if is_goal(item):
            return True
        heappush(fringe, (counter, item))

    prev_city = initial_city

    while len(fringe) > 0:

        state = heappop(fringe)
        heappush(fringe, (0, state[1]))
        temp = state[1]

        if temp[0] == prev_city:
            pre_city = temp[1]
        else:
            pre_city = temp[0]

        counter -= 1

        if pre_city not in visited_cities:
            if cost == "distance":
                for succ in successors_astar_distance(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
            if cost == "time":
                for succ in successors_astar_time(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
            if cost == "segments":
                for succ in successors_dfs(state, pre_city):
                    if is_goal(succ):
                        return True

                    elif succ[0] not in visited_cities and succ[1] not in visited_cities:
                        heappush(fringe, (counter, succ))
                        c += 1
        visited_cities.append(pre_city)
        prev_city = pre_city

    return False


def solve_bfs(initial_city, cost):
    city_tuple = []
    c = 0
    for item in road_segments:
        if item[0] == initial_city or item[1] == initial_city:
            city_tuple.append(item)
    fringe = city_tuple
    while len(fringe) > 0:
        state = fringe.pop()
        if cost == "distance":
            for succ in successors_bfs_distance(state):
                if is_goal(succ):
                    return True
                else:
                    fringe.insert(0, succ)
        if cost == "time":
            for succ in successors_bfs_time(state):
                if is_goal(succ):
                    return True
                else:
                    fringe.insert(0, succ)
        if cost == "segments":
            for succ in successors_bfs(state):
                if is_goal(succ):
                    return True
                else:
                    fringe.insert(0, succ)
                    c += 1
    return False


def solve_uniform(initial_city):
    city_tuple = []
    for item in road_segments:
        if item[0] == initial_city or item[1] == initial_city:
            city_tuple.append(item)
    fringe = []
    for item in city_tuple:
        heappush(fringe, (item[2], item))
    while len(fringe) > 0:

        state = heappop(fringe)
        for succ in successors_uniform(state):
            if is_goal(succ):
                print("Found using Uniform")
                return True
            else:
                heappush(fringe, (succ[0] + state[0], succ))
    return False

def solve(start, end, algorithm, cost):
    if algorithm == "bfs":
        solve_bfs(start, cost)
    if algorithm == "dfs":
        solve_dfs(start, cost)
    if algorithm == "uniform":
        solve_uniform(start, cost)
    if algorithm == "ids":
        solve_ids(start, cost)
    if algorithm == "astar":
        solve_astar(start, cost)
    return

road_segments = []
with open(path, 'r') as f:
    for line in f:
        road_segments.append(tuple(line.rstrip().split()))

start_city = sys.argv[1]
destination_city = sys.argv[2]
routing_algorithm = sys.argv[3]
cost_function = sys.argv[4]