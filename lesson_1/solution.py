import os, sys, time, math

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.ai_lab_functions import *
import gym, envs

def BFS_TreeSearch(problem):
    """
    Tree Search BFS
    
    Args:
        problem: OpenAI Gym environment
        
    Returns:
        (path, time_cost, space_cost): solution as a path and stats.
    """
    
    node = Node(problem.startstate, None)
    time_cost = 0
    space_cost = 1
    frontier = NodeQueue()
    frontier.add(node)
    found = False # found the solution
    while not frontier.is_empty() and not found:
        current = frontier.remove()
        for move in range(problem.action_space.n):
            time_cost += 1
            new_node = Node(problem.sample(current.state,move),current)
            if(new_node.state == problem.goalstate):
                node = new_node
                found = True
                break
            else:
                frontier.add(new_node)
        if(len(frontier)>space_cost):
            space_cost = len(frontier)
    return build_path(node), time_cost, space_cost

def BFS_GraphSearch(problem):
    """
    Graph Search BFS
    
    Args:
        problem: OpenAI Gym environment
        
    Returns:
        (path, time_cost, space_cost): solution as a path and stats.
    """
    
    node = Node(problem.startstate, None)
    time_cost = 0
    space_cost = 1
    #
    # YOUR CODE HERE ...
    #
    return build_path(node), time_cost, space_cost  

if __name__=="__main__":
    envname = "SmallMaze-v0"
    environment = gym.make(envname)

    solution_ts, time_ts, memory_ts = BFS_TreeSearch(environment)
    #solution_gs, time_gs, memory_gs = BFS_GraphSearch(environment)

    print("\n----------------------------------------------------------------")
    print("\tBFS TREE SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Solution: {}".format(solution_2_string(solution_ts, environment)))
    print("N° of nodes explored: {}".format(time_ts))
    print("Max n° of nodes in memory: {}".format(memory_ts))

    #print("\n----------------------------------------------------------------")
    #print("\tBFS GRAPH SEARCH PROBLEM: ")
    #print("----------------------------------------------------------------")
    #print("Solution: {}".format(solution_2_string(solution_gs, environment)))
    #print("N° of nodes explored: {}".format(time_gs))
    #print("Max n° of nodes in memory: {}".format(memory_gs))