from bucket_elimination import BucketElimination
from bucket import Bucket


def constraint_partitioning(bucket_elimination, variable_order, soft_constraints, hard_constraints):
    """
        Generate the bucket with the corresponding constraints in the correct order (inverse of the given), and add all the buckets to the bucket_elimination object that represent the problem.

        Parameters
        ----------
            bucket_elimination : BucketElimination
                the object of the class BucketElimination that represent the current problem (empty).
            variable_order : list
                the variables that appear in the problem in the given order.
            soft_constraints : list
                the soft contraints, a list of lists, each list is built with the function name for the first element, followed by the intereseted variables.
            hard_constraints : list
                the hard contraints (only inequality constraints), a list of lists, each list represent the variable interested in the inequality contraints.

        Returns:
        --------
            bucket_elimination : BucketElimination
                the object of the class BucketElimination that represents the current problem (with the bucket filled).
    """
    already_added = []
    for var in reversed(variable_order):
        soft = []
        hard = []
        for const in hard_constraints:
            if const not in already_added and var in const:
                hard.append(const)
                already_added.append(const)
        for const in soft_constraints:
            if const not in already_added and var in const:
                soft.append(const)
                already_added.append(const)
        bucket = Bucket(var,soft,hard)
        bucket_elimination.add(bucket)
    return bucket_elimination


def main_bucket_elimination(problem_name, problem_definition):
    """
        Main script of the bucket elimination, given the problem definition compute the global_maximum,
        the correct assignment and the memory cost of the process.

        Parameters
        ----------
            problem_name : str
                the name of the problem, for visualization purpose.
            problem_definition : list
                complete definition of the problem, a list that contain (in order):
                problem_domains, variable_order, problem_soft_constraints and problem_hard_constraints.
        """
    # Extract the problem constant from the parameter "problem_definition"
    problem_domains, problem_order, problem_soft_constraints, problem_hard_constraints = problem_definition
    bucket_elimination = constraint_partitioning(BucketElimination(problem_domains),problem_order,problem_soft_constraints,problem_hard_constraints)

    bucket_elimination.bucket_processing()
    for table in bucket_elimination.get_tables():
        print()
        Bucket.plot_table(table)

    assignment,global_maximum = bucket_elimination.value_propagation()

    evaluations = evaluate_soft_constraints(assignment,problem_soft_constraints)

    max_table_size = get_max_table_size(bucket_elimination.get_tables())
    # Plot all the computed results
    print(f"\nBucket Elimination for the: {problem_name}:")
    print(f"\tVariable Assignment: {assignment}")
    print(f"\tGlobal Maximum Found: {global_maximum}")
    print(f"\tMaximum Table Size (with the order {problem_order}): {max_table_size}")
    print("\tGraphical Visualization:")
    #print(evaluations)
    bucket_elimination.plot_assignment_as_graph(assignment, evaluations)


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
    for table in final_tables:
        max_table_size = max(max_table_size,len(table[0])*len(table))
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
        function = elem[0]
        val = function(assignment[elem[1]],assignment[elem[2]])
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
    main_bucket_elimination("Partial Test 15/05/2013", PROBLEM_1)