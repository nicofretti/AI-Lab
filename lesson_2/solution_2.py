import os
import sys
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.ai_lab_functions import *
import gym
import envs

def present_with_higher_cost(queue, node):
    if (node.state in queue):
        if(queue[node.state].value > node.value): return True
    return False

def astar_tree_search(environment):
    goalpos = environment.state_to_pos(environment.goalstate)
    queue = PriorityQueue()
    queue.add(Node(environment.startstate))
    time_cost = 0
    space_cost = 1
    while True:
        if (queue.is_empty()):
            return None
        node = queue.remove()
        if (node.state == environment.goalstate):
            return build_path(node), time_cost, space_cost
        for action in range(environment.action_space.n):
            child_state = environment.sample(node.state, action)
            euristic = Heu.l1_norm(environment.state_to_pos(child_state), goalpos)
            child = Node(child_state, node, node.pathcost + 1, euristic + node.pathcost)
            time_cost += 1
            queue.add(child)
            space_cost = max(space_cost, len(queue))
    return None, time_cost, space_cost


def astar_graph_search(environment):
    goalpos = environment.state_to_pos(environment.goalstate)
    queue = PriorityQueue()
    queue.add(Node(environment.startstate))
    time_cost = 0
    space_cost = 1
    explored = set()
    while True:
        if(queue.is_empty()):
            return None
        node = queue.remove()
        if(node.state==environment.goalstate):
            return build_path(node),time_cost,space_cost
        explored.add(node.state)
        for action in range(environment.action_space.n):
            child_state = environment.sample(node.state,action)
            euristic = Heu.l1_norm(environment.state_to_pos(child_state),goalpos)
            child = Node(child_state, node, node.pathcost+1, euristic + node.pathcost)
            time_cost+=1
            if(child.state not in explored and child.state not in queue):
                queue.add(child)
            elif present_with_higher_cost(queue, child):
                queue.replace(child)
            space_cost = max(space_cost,len(queue)+len(explored))
    return None, time_cost, space_cost


def astar(environment, search_type):
    path, time_cost, space_cost = search_type(environment)
    return path, time_cost, space_cost

if __name__=="__main__":
    envname = "SmallMaze-v0"
    environment = gym.make(envname)

    solution_ts, time_ts, memory_ts = astar(environment, astar_tree_search)
    solution_gs, time_gs, memory_gs = astar(environment, astar_graph_search)

    print("\n----------------------------------------------------------------")
    print("\tA* TREE SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Solution: {}".format(solution_2_string(solution_ts, environment)))
    print("N° of nodes explored: {}".format(time_ts))
    print("Max n° of nodes in memory: {}".format(memory_ts))

    print("\n----------------------------------------------------------------")
    print("\tA* GRAPH SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Solution: {}".format(solution_2_string(solution_gs, environment)))
    print("N° of nodes explored: {}".format(time_gs))
    print("Max n° of nodes in memory: {}".format(memory_gs))