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


def ucs(environment):
    queue = PriorityQueue()
    queue.add(Node(environment.startstate))

    explored = set()
    time_cost = 0
    space_cost = 1

    while True:
        if queue.is_empty(): return None

        node = queue.remove()  # Retrieve node from the queue
        if node.state == environment.goalstate: return build_path(node), time_cost, space_cost
        explored.add(node.state)

        for action in range(environment.action_space.n):  # Look around
            # Child node where value and pathcost are both the pathcost of parent + 1
            child = Node(environment.sample(node.state, action), node, node.pathcost + 1, node.pathcost + 1)
            time_cost += 1
            if (child.state not in queue and child.state not in explored):
                # if child.state == environment.goalstate: return build_path(child), time_cost, space_cost
                queue.add(child)
            elif present_with_higher_cost(queue, child):
                queue.replace(child)
        space_cost = max(space_cost, len(queue) + len(explored))


def greedy_tree_search(environment, timeout=10000):
    goalpos = environment.state_to_pos(environment.goalstate)
    queue = PriorityQueue()
    queue.add(Node(environment.startstate))
    time_cost = 0
    space_cost = 1
    while True:
        if time_cost >= timeout: return ("time-out", time_cost, space_cost)  # timeout check
        if(queue.is_empty()):
            return None
        node = queue.remove()
        if(node.state==environment.goalstate):
            return build_path(node),time_cost,space_cost
        for action in range(environment.action_space.n):
            child_state = environment.sample(node.state,action)
            euristic = Heu.l1_norm(environment.state_to_pos(child_state),goalpos)
            child = Node(child_state, node, node.pathcost+1, euristic)
            time_cost+=1
            queue.add(child)
            space_cost = max(space_cost,len(queue))
    return None, time_cost, space_cost


def greedy_graph_search(environment):
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
            child = Node(child_state, node, node.pathcost+1, euristic)
            time_cost+=1
            if(child.state not in explored and child.state not in queue):
                queue.add(child)
            space_cost = max(space_cost,len(queue)+len(explored))
    return None, time_cost, space_cost

def greedy(environment, search_type):
    path, time_cost, space_cost = search_type(environment)
    return path, time_cost, space_cost

if __name__=="__main__":
    envname = "SmallMaze-v0"
    environment = gym.make(envname)

    solution_ts, time_ts, memory_ts = greedy(environment, greedy_tree_search)
    solution_gs, time_gs, memory_gs = greedy(environment, greedy_graph_search)

    print("\n----------------------------------------------------------------")
    print("\tGREEDY BEST FIRST TREE SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Solution: {}".format(solution_2_string(solution_ts, environment)))
    print("N° of nodes explored: {}".format(time_ts))
    print("Max n° of nodes in memory: {}".format(memory_ts))

    print("\n----------------------------------------------------------------")
    print("\tGREEDY BEST FIRST GRAPH SEARCH PROBLEM: ")
    print("----------------------------------------------------------------")
    print("Solution: {}".format(solution_2_string(solution_gs, environment)))
    print("N° of nodes explored: {}".format(time_gs))
    print("Max n° of nodes in memory: {}".format(memory_gs))