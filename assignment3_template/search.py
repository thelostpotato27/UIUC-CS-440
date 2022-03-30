# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.
from collections import deque
from heapq import heappop, heappush
class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): DISTANCE(i, j)
                for i, j in self.cross(objectives)
            }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    maze_bfs = deque()
    starting_tuple = (maze.start , "blank")
    maze_bfs.append(starting_tuple)
    visited_points = [[0 for x in range(maze.size.x)] for y in range(maze.size.y)]

    while maze_bfs[0][0] != maze.waypoints[0]:
        #print(len(maze_bfs))
        # visited_sum = 0
        # for elems in visited_points:
        #     visited_sum += sum(elems)


        neighbors = maze.neighbors(maze_bfs[0][0][0], maze_bfs[0][0][1])
        for elem in neighbors:
            if visited_points[elem[0]][elem[1]] == 1:
                continue

            maze_bfs.append((elem, maze_bfs[0]))
            visited_points[elem[0]][elem[1]] = 1
        maze_bfs.popleft()

    answer = []
    maze_bfs = maze_bfs[0]
    while maze_bfs[0] != maze.start:
        answer.append(maze_bfs[0])
        maze_bfs = maze_bfs[1]

    answer.append(maze_bfs[0])
    answer.reverse()
    return answer

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    maze_A_star = []
    heappush(maze_A_star, (2*maze.size.x + 2*maze.size.y, 0, (start)))
    start = maze.start
    finish = maze.waypoints[0]
    visited_points = [[0 for x in range(maze.size.x)] for y in range(maze.size.y)]
    distance = abs(start[0] - finish[0][0]) + abs(start[1] - finish[0][1])
    heappush(maze_A_star, (distance, 0, (start)))
    while maze_A_star[0][0] != maze_A_star[0][1]:
        neighbors = maze.neighbors(maze_A_star[0][2][0][0], maze_A_star[0][2][0][0])
        visited_points[maze_A_star[0][2][0][0]][maze_A_star[0][2][0][1]] = 1
        trail = maze_A_star[0][2]
        current_location = heappop(maze_A_star)
        for elem in neighbors:
            if visited_points[elem[0]][elem[1]] == 1:
                continue
            traveled = current_location[1] + 1
            distance = abs(elem[0] - finish[0][0]) + abs(elem[1] - finish[0][1]) + traveled
            heappush(maze_A_star, (distance, traveled, (elem, trail)))
    answer = []
    maze_A_star = maze_A_star[0][2]
    while maze_A_star[0] != start:
        answer.append(maze_A_star[0])
        maze_A_star = maze_A_star[1]

    answer.append(maze_A_star[0])
    answer.reverse()
    return answer


def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []


