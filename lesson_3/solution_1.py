from bucket_elimination import BucketElimination
from bucket import Bucket


def constraint_partitioning(bucket_elimination, variable_order, soft_constraints, hard_constraints):
    already_added = []
    for var in reversed(variable_order):
        soft = []
        hard = []
        for constraint in hard_constraints:
            if constraint not in already_added and var in constraint:
                hard.append(constraint)
        for constraint in soft_constraints:
            if constraint not in already_added and var in constraint:
                soft.append(constraint)
        bucket = Bucket(var,soft,hard)
        bucket_elimination.add(bucket)
    return bucket_elimination


def main_bucket_elimination(problem_name, problem_definition):

    # Extract the problem constant from the parameter "problem_definition"
    problem_domains, problem_order, problem_soft_constraints, problem_hard_constraints = problem_definition
    assignment, global_maximum, max_table_size = None, None, None
    bucket_elimination = constraint_partitioning(BucketElimination(problem_domains),problem_order,problem_soft_constraints,problem_hard_constraints)

    bucket_elimination.bucket_processing()
    for table in bucket_elimination.get_tables():
        print()
        Bucket.plot_table(table)

    assignment,global_maximum = bucket_elimination.value_propagation()

    evaluations = evaluate_soft_constraints(assignment,problem_soft_constraints)

    max_table_size = get_max_table_size(bucket_elimination.get_tables())
    # Plot all the computed results
    #print(f"\nBucket Elimination for the: {problem_name}:")
    #print(f"\tVariable Assignment: {assignment}")
    #print(f"\tGlobal Maximum Found: {global_maximum}")
    #print(f"\tMaximum Table Size (with the order {problem_order}): {max_table_size}")
    #print("\tGraphical Visualization:")
    #print(evaluations)
    #bucket_elimination.plot_assignment_as_graph(assignment, evaluations)


def get_max_table_size(final_tables):
    """
    Compute the maximum number of elements that appear in one of the table generated inside the main process.

    Parameters
    ----------
        final_tables : list
            list of the tables generated inside the loop for each bucket.

    Returns:
    --------
        max_table_size : int
            the number of elements inside the largest table (i.e., number of row multiplied by the number of columns).
    """

    # Variable initialization
    max_table_size = 0
    current_lenght = 0
    for table in final_tables:
        current_lenght = 0
        for row in table:
            current_lenght += len(row)
        if(current_lenght>max_table_size):
            max_table_size = current_lenght
    print(max_table_size)
    return max_table_size


def evaluate_soft_constraints(assignment, soft_constraints):
    """
    Compute the value of the soft constraints, evaluating them on the given the variables assignment.

    Parameters
    ----------
        assignment : list
            the assignment for each variable to obtain the maximum (the key is the literal and the value is the assigned value).
        soft_constraints : list
            the soft contraints, a list of lists, each list is built with the function name for the first element, followed by the intereseted variables.

    Returns:
    --------
        evaluations : list
            a list with the results of the evaluation of the soft constraints given the variables assignment.
    """

    # Variable initialization
    evaluations = []
    for elem in soft_constraints:
        val = elem[0](assignment[elem[1]],assignment[elem[2]])
        evaluations.append(val)
    return evaluations

def F_1(x_i, x_j):
    if x_i != x_j:
        return 0
    elif x_i == 'R' and x_j == 'R':
        return -1
    elif x_i == 'B' and x_j == 'B':
        return -2
    else:
        raise ValueError("Invalid Value for F")

def F_2(x_i, x_j):
    if x_i != x_j:
        return 0
    elif x_i == 'R' and x_j == 'R':
        return 2
    elif x_i == 'B' and x_j == 'B':
        return 1
    else:
        raise ValueError("Invalid Value for F")

PROBLEM_GC = [
    {'X1': ['R', 'B', 'Y'], 'X2': ['R', 'B', 'Y'], 'X3': ['R', 'B', 'Y'], 'X4': ['R', 'B', 'Y'], 'X5': ['R', 'B', 'Y']},
    # PROBLEM DOMAINS
    ['X5', 'X4', 'X3', 'X2', 'X1'],  # PROBLEM ORDER
    [],  # PROBLEM SOFT CONSTRAINTS
    [['X1', 'X2'], ['X2', 'X3'], ['X3', 'X4'], ['X2', 'X4'], ['X1', 'X4'], ['X2', 'X5'], ['X3', 'X5'], ['X1', 'X5']]
    # PROBLEM HARD CONSTRAINTS
]

PROBLEM_2 = [
    {'X1': ['R', 'B'], 'X2': ['R', 'B'], 'X3': ['R', 'B'], 'X4': ['R', 'B']},  # PROBLEM DOMAINS
    ['X1', 'X2', 'X3', 'X4'],  # PROBLEM ORDER
    [[F_2, 'X1', 'X2'], [F_2, 'X2', 'X3'], [F_2, 'X2', 'X4']],  # PROBLEM SOFT CONSTRAINTS
    [['X1', 'X3'], ['X3', 'X4']]  # PROBLEM HARD CONSTRAINTS
]

PROBLEM_1 = [
    {'X1': ['R', 'B'], 'X2': ['R', 'B'], 'X3': ['R', 'B'], 'X4': ['R', 'B']},  # PROBLEM DOMAINS
    ['X4', 'X3', 'X2', 'X1'],  # PROBLEM ORDER
    [[F_1, 'X1', 'X2'], [F_1, 'X1', 'X4'], [F_1, 'X2', 'X4'], [F_1, 'X3', 'X4']],[]]  # PROBLEM SOFT CONSTRAINTS

if __name__=="__main__":
    main_bucket_elimination("Problem Graph Coloring", PROBLEM_GC)