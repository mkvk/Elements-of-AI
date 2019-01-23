#!/bin/python
# put your group assignment problem here!
# Code written in python3

# The approach goes as follows:
# 1. Take the sample text input file
# 2. Take the text file and break it as a list of tuples -- groups
# 3. Randomly assign each group and check the time for that assignment
# 4. Repeat the procedure for 10,000 iterations
# 5. Find the minimum time taken from these 10,000 group assignments
# 6. Print the minimum time group assignment along with the time

import sys
import random

# Path takes the text input file
path = sys.argv[1]

# K is the number of minutes it takes for grading an assignment submitted by each group
k = int(sys.argv[2])

# M is the number of minutes it takes for every meeting when the student is assigned a person he doesn't want to work with
m = int(sys.argv[3])

# N is the number of minutes it takes when a student is not assigned a person he wants to work with
n = int(sys.argv[4])


# This function calculates the time taken for a group assignment
def time(assigned, groups):

    # Initialize a dictionary with keys as the unique student IDs and values as the tuple for that student
    group_dict = {}

    # Group size conflict time is 1 for each such case
    group_size_conflict_time = 0

    # want to work with conflict time is N times each such case
    want_to_work_with_conflict_time = 0

    # not to work with conflict time is M times each such case
    not_to_work_with_conflict_time = 0

    # The grading time is the k times number of groups in the group assignment
    grade_time = len(assigned) * k

    # Creating the groups dictionary with keys as student IDs and values as their preferences
    for item in groups:
        group_dict[item[0]] = item

    # Start iterating for every group in the group assignment
    for team in assigned:

        # First we check the group size conflicts
        size = len(team)

        # For every person in one team -- iterates for the size of the team
        for x in range(size):
            first = team[x]

            # Retrieving the preferences for the specified person in team
            group_ele = group_dict[first]

            # Checking if the group preference matches with the assigned group
            if int(group_ele[1]) == len(team) or group_ele[1] == '0':
                continue
            else:
                group_size_conflict_time += 1

            # There may be multiple persons in the want to work with preference, so take a list for it
            want_to_work_with = []

            # There may be multiple persons in the do not want to work with preference, so take a list for it
            not_to_work_with = []

            # want to work with preference is the 3rd element of the preference tuple
            a = group_ele[2]

            # do not want to work with preference is the 4th element of the preference tuple
            b = group_ele[3]

            # When '_' is given it means no preference, so we don't consider for that
            # Otherwise we split them and put each of them into a list for comparison
            if a != '_':
                want_to_work_with = a.split(',')

            # When '_' is given it means no preference, so we don't consider for that
            # Otherwise we split them and put each of them into a list for comparison
            if b != '_':
                not_to_work_with = b.split(',')

            # If the list if not empty, then we check
            if len(want_to_work_with) > 0:

                # For every item in the want to work with preference, check if that person is there in the assigned team
                # If the person is not in the assigned team, add the N conflict time
                for per in want_to_work_with:
                    if per not in team:
                        want_to_work_with_conflict_time += n

            # If the list if not empty, we check
            if len(not_to_work_with) > 0:

                # For every item in the do not want to work with preference, check if that person is there in the assigned team
                # If the person is in the assigned team, add the M conflict time
                for per in not_to_work_with:
                    if per in team:
                        not_to_work_with_conflict_time += m

    # Finally, calculate the total time taken for each group in the group assignment
    # Total = k + m + n
    total = group_size_conflict_time + not_to_work_with_conflict_time + want_to_work_with_conflict_time + grade_time
    return total

# This function picks a random group assignment for the sample imput text file
def group_assign(teams, students):

    # Assign is the list which contains all the groups for the group assignment
    assign = []

    # Students_alias is a list used to avoid duplicates in the groups
    students_alias = students

    # For every student in the sample text file
    for team in teams:

        # This is the ID of the student according to whose preferences the group is being assigned
        student_ID = team[0]

        # Checking the team size preference of the student
        # Randomly assign 2 people from the student_alias list to the current student
        # and delete these three from the student_alias list
        # The random.choice() throws an indexerror, so have to check if the student_alias list is not empty for each call
        if team[1] == '3':
            if team[0] not in students_alias:
                break
            else:
                if len(students_alias) > 0:
                    students_alias.remove(student_ID)

                if len(students_alias) > 0:
                    a = random.choice(students_alias)
                if len(students_alias) > 0:
                    students_alias.remove(a)

                if len(students_alias) > 0:
                    b = random.choice(students_alias)
                if len(students_alias) > 0:
                    students_alias.remove(b)

                # Append the group to the assign list
                assign.append(tuple([student_ID, a, b]))

        # Checking the team size preference of the student
        # Randomly assign 1 person from the student_alias list to the current student
        # and delete these two from the student_alias list
        # The random.choice() throws an indexerror, so have to check if the student_alias list is not empty for each call
        if team[1] == '2':
            if team[0] not in students_alias:
                break
            else:
                if len(students_alias) > 0:
                    students_alias.remove(student_ID)
                if len(students_alias) > 0:
                    a = random.choice(students_alias)
                if len(students_alias) > 0:
                    students_alias.remove(a)

                # Append the group to the assign list
                assign.append(tuple([student_ID, a]))

        # Checking the team size preference of the student
        # Let the person stay alone in the group
        # delete this person from the student_alias list
        # The random.choice() throws an indexerror, so have to check if the student_alias list is not empty for each call
        if team[1] == '1':
            if team[0] in students_alias:
                if len(students_alias) > 0:
                    students_alias.remove(student_ID)

                # Append the group to the assign list
                assign.append(tuple([student_ID]))

    # There may be some cases when the students with group size as '0' are left alone without any group
    # This means that there are still students left in the student_alias list
    # Now, we group these students first in groups of 3, then 2, and then 1

    while len(students_alias) > 0:

        # Grouping students in groups of 3
        if len(students_alias) >= 3:
            first = students_alias[0]
            if len(students_alias) > 0:
                students_alias.remove(first)
            if len(students_alias) > 0:
                a = random.choice(students_alias)
            if len(students_alias) > 0:
                students_alias.remove(a)
            if len(students_alias) > 0:
                b = random.choice(students_alias)
            if len(students_alias) > 0:
                students_alias.remove(b)

            # Append the group to the assign list
            assign.append(tuple([first, a, b]))

        # After grouping into teams of 3, the remaining are grouped into teams of 2 or 1
        if len(students_alias) == 2:
            first = students_alias[0]
            if len(students_alias) > 0:
                students_alias.remove(first)
            if len(students_alias) > 0:
                a = random.choice(students_alias)
            if len(students_alias) > 0:
                students_alias.remove(a)

            # Append the group to the assign list
            assign.append(tuple([first, a]))

        # Grouping into teams of 1
        if len(students_alias) == 1:
            first = students_alias[0]
            if len(students_alias) > 0:
                students_alias.remove(first)

            # Append the group to the assign list
            assign.append(tuple([first]))
    return assign

# The main function of the program
def main():

    # This is the list which stores each group assignment from an iteration
    random_teams = []

    # This is the list which stores the time for each group assignment simultaneously
    min_total = []

    # Minimum is the minimum time taken from all the group assignments in 10,000 iterations
    minimum = 0

    # Begin the iterations
    for i in range(10000):

        # Groups is every line taken as tuple from the sample text input file
        groups = []

        # Students is the unique student ID from the sample text input file
        students = []

        # Group_no is the group number of every student from the sample text input file
        group_no = []

        # Take the input file and split it as a list of tuples
        with open(path, 'r') as f:
            for line in f:
                groups.append(tuple(line.rstrip().split()))

        # The students list
        for item in groups:
            students.append(item[0])

        # The Group_no list
        for item in groups:
            group_no.append(item[1])

        # Doing a random group assignment
        a = group_assign(groups, students)

        # Appending each group assignment to random_teams
        random_teams.append(a)

        # Calculating the time for each group assignment
        b = time(a, groups)

        # Appending the time for each group assignment to min_total
        min_total.append(b)

    # Extracting the minimum time taken for a group assignment
    minimum = min(min_total)

    # Extracting the group assignment of the minimum time
    index = min_total.index(minimum)
    optimal = random_teams[index]

    # Printing the minimum time group assignment
    for item in optimal:
        print(*item, sep = " ")
    print(minimum)
    return

main()


