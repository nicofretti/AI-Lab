import os, sys, time, math

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.ai_lab_functions import *
import gym, envs


def DLS(problem, limit, RDLS_Function):
    node = Node(problem.startstate, None)
    return RDLS_Function(node, problem, limit, set())

def IDS(problem, DLS_Function):
    """
    Iteartive_DLS DLS
    Args: problem=OpenAI Gym environment
    Returns: (path, time_cost, space_cost): solution as a path and stats.
    """
    total_time_cost = 0
    total_space_cost = 1
    for i in zero_to_infinity():
        result,time_cost,space_cost = DLS(problem,i,DLS_Function)
        #print("limit:{}-result:{}".format(i,result))
        total_space_cost = max(total_space_cost, space_cost)
        total_time_cost += time_cost
        if(result!='cut_off'):
            return result,total_time_cost,total_space_cost,i

def Recursive_DLS_TreeSearch(node, problem, limit, explored):
    """
    Recursive DLS
    Args:   node: node to explore
            problem: OpenAI Gym environment
            limit: depth limit for the exploration, negative number means 'no limit'
            explored: completely explored nodes
    Returns:(path, time_cost, space_cost): solution as a path and stats.
    """
    time_cost = 1
    space_cost = node.pathcost
    #print("test:{}-goal:{}".format(problem.state_to_pos(problem.goalstate),problem.state_to_pos(node.state)))
    if problem.goalstate == node.state:
        return build_path(node), time_cost, len(explored)
    if(limit==0):
        return "cut_off", time_cost, space_cost
    cut_off_occurred = False
    for move in range(problem.action_space.n):
        child = Node(problem.sample(node.state,move),node,node.pathcost+1)
        result,under_time,under_space = Recursive_DLS_TreeSearch(child,
                                                                  problem,
                                                                  limit-1,
                                                                  explored)
        time_cost += under_time
        space_cost = max(space_cost, under_space)
        if result == 'cut_off':
            cut_off_occurred = True
        if result != 'failure' and result!='cut_off':
            return result,time_cost,space_cost
    if cut_off_occurred:
        return 'cut_off',time_cost, space_cost
    return 'failure', time_cost, space_cost


def Recursive_DLS_GraphSearch(node, problem, limit, explored):
    """
    Recursive DLS
    Args:   node: node to explore
            problem: OpenAI Gym environment
            limit: depth limit for the exploration, negative number means 'no limit'
            explored: completely explored nodes
    Returns:(path, time_cost, space_cost): solution as a path and stats.
    """
    #print("test:{}-goal:{}".format(problem.state_to_pos(problem.goalstate),problem.state_to_pos(node.state)))
    if problem.goalstate == node.state:
        return build_path(node), 1, node.pathcost
    if(limit==0):
        return "cut_off", 1, node.pathcost
    time_cost = 1
    space_cost = node.pathcost
    explored.add(node.state)
    cut_off_occurred = False
    for move in range(problem.action_space.n):
        child = Node(problem.sample(node.state,move),node,node.pathcost+1)
        if(not child.state in explored):
            result,under_time,under_space = Recursive_DLS_GraphSearch(child,
                                                                      problem,
                                                                      limit-1,
                                                                      explored)
            time_cost += under_time
            space_cost = max(space_cost, under_space)
            if result == 'cut_off':
                cut_off_occurred = True
            if result != 'failure' and result!='cut_off':
                return result,time_cost,space_cost
    if cut_off_occurred:
        return 'cut_off',time_cost, space_cost
    return 'failure', time_cost, space_cost

if __name__=="__main__":
    envname = "SmallMaze-v0"
    environment = gym.make(envname)

    solution_ts, time_ts, memory_ts, iterations_ts = IDS(environment, Recursive_DLS_TreeSearch)
    solution_gs, time_gs, memory_gs, iterations_gs = IDS(environment, Recursive_DLS_GraphSearch)

    print("\n----------------------------------------------------------------")
    print("\tIDS TREE SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Necessary Iterations: {}".format(iterations_ts))
    print("Solution: {}".format(solution_2_string(solution_ts, environment)))
    print("N° of nodes explored: {}".format(time_ts))
    print("Max n° of nodes in memory: {}".format(memory_ts))

    print("\n----------------------------------------------------------------")
    print("\tIDS GRAPH SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Necessary Iterations: {}".format(iterations_gs))
    print("Solution: {}".format(solution_2_string(solution_gs, environment)))
    print("N° of nodes explored: {}".format(time_gs))
    print("Max n° of nodes in memory: {}".format(memory_gs))