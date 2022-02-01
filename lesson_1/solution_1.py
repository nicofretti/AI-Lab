import os, sys, time, math

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.ai_lab_functions import *
import gym, envs

def BFS_TreeSearch(problem):
    """
    Tree Search BFS
    Args->problem: OpenAI Gym environment
    Returns->(path, time_cost, space_cost): solution as a path and stats.
    """
    node = Node(problem.startstate, None)
    time_cost = 0
    space_cost = 1

    if node.state == problem.goalstate:
        return build_path(node), time_cost, space_cost

    frontier = NodeQueue()
    frontier.add(node)

    while not frontier.is_empty():
        current = frontier.remove()

        for action in range(problem.action_space.n):
            time_cost += 1
            child = Node(problem.sample(current.state, action), current)
            if(child.state == problem.goalstate):
                return build_path(child), time_cost, space_cost # solution
            frontier.add(child)

        space_cost = max(space_cost,len(frontier))
    return None, time_cost, space_cost #failure

def BFS_GraphSearch(problem):
    """
    Graph Search BFS
    Args->problem: OpenAI Gym environment
    Returns->(path, time_cost, space_cost): solution as a path and stats.
    """
    node = Node(problem.startstate, None)
    time_cost = 0
    space_cost = 1

    if node.state == problem.goalstate:
        return build_path(node), time_cost, space_cost

    frontier = NodeQueue()
    explored = []
    frontier.add(node)

    while not frontier.is_empty():
        current = frontier.remove()
        explored.append(current.state)
        for action in range(problem.action_space.n):
            child = Node(problem.sample(current.state, action), current)
            if (child.state not in explored) and (child.state not in frontier):
                if problem.goalstate == child.state:
                    return build_path(child), time_cost, space_cost #solution
                frontier.add(child)
            time_cost += 1
        space_cost = max(space_cost, len(explored)+len(frontier))
    return None, time_cost, space_cost #failure

if __name__=="__main__":
    envname = "SmallMaze-v0"
    environment = gym.make(envname)

    solution_ts, time_ts, memory_ts = BFS_TreeSearch(environment)
    solution_gs, time_gs, memory_gs = BFS_GraphSearch(environment)

    print("\n----------------------------------------------------------------")
    print("\tBFS TREE SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Solution: {}".format(solution_2_string(solution_ts, environment)))
    print("N° of nodes explored: {}".format(time_ts))
    print("Max n° of nodes in memory: {}".format(memory_ts))

    print("\n----------------------------------------------------------------")
    print("\tBFS GRAPH SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Solution: {}".format(solution_2_string(solution_gs, environment)))
    print("N° of nodes explored: {}".format(time_gs))
    print("Max n° of nodes in memory: {}".format(memory_gs))