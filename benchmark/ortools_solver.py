"""
Google OR-Tools solver for Set Cover Problem.
Solves both linear relaxation and integer programming formulation.
"""

from ortools.linear_solver import pywraplp
import time


def solve_scp_linear_relaxation(problem_data):
    """
    Solve SCP linear relaxation using OR-Tools.

    Args:
        problem_data: dict with 'num_sets', 'num_elements', 'costs', 'sets'

    Returns:
        dict with solution info: {
            'status': str,
            'objective': float,
            'primal_solution': list,
            'dual_solution': list (constraint duals),
            'solve_time': float
        }
    """
    num_sets = problem_data["num_sets"]
    num_elements = problem_data["num_elements"]
    costs = problem_data["costs"]
    sets = problem_data["sets"]

    # Create solver (use GLOP for LP)
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return {
            "status": "SOLVER_UNAVAILABLE",
            "objective": None,
            "primal_solution": [],
            "dual_solution": [],
            "solve_time": 0.0,
        }

    # Decision variables: x[i] = fraction of set i selected (continuous 0-1)
    x = [solver.NumVar(0.0, 1.0, f"x_{i}") for i in range(num_sets)]

    # Objective: minimize sum of costs
    objective = solver.Objective()
    for i in range(num_sets):
        objective.SetCoefficient(x[i], costs[i])
    objective.SetMinimization()

    # Constraints: each element must be covered at least once
    constraints = []
    for elem in range(1, num_elements + 1):
        constraint = solver.Constraint(1.0, solver.infinity(), f"cover_{elem}")
        for set_idx, s in enumerate(sets):
            if elem in s:
                constraint.SetCoefficient(x[set_idx], 1.0)
        constraints.append(constraint)

    # Solve
    start_time = time.time()
    status = solver.Solve()
    solve_time = time.time() - start_time

    # Extract solution
    status_name = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status, "UNKNOWN")

    primal_solution = [x[i].solution_value() for i in range(num_sets)]
    dual_solution = [constraints[i].dual_value() for i in range(len(constraints))]

    return {
        "status": status_name,
        "objective": (
            solver.Objective().Value()
            if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]
            else None
        ),
        "primal_solution": primal_solution,
        "dual_solution": dual_solution,
        "solve_time": solve_time,
    }


def solve_scp_integer(problem_data, time_limit_sec=300):
    """
    Solve SCP as integer program using OR-Tools.

    Args:
        problem_data: dict with 'num_sets', 'num_elements', 'costs', 'sets'
        time_limit_sec: time limit in seconds

    Returns:
        dict with solution info
    """
    num_sets = problem_data["num_sets"]
    num_elements = problem_data["num_elements"]
    costs = problem_data["costs"]
    sets = problem_data["sets"]

    # Create solver (use SCIP for MIP)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return {
            "status": "SOLVER_UNAVAILABLE",
            "objective": None,
            "solution": [],
            "solve_time": 0.0,
            "gap": None,
        }

    solver.SetTimeLimit(time_limit_sec * 1000)  # milliseconds

    # Decision variables: x[i] = 1 if set i is selected, 0 otherwise
    x = [solver.BoolVar(f"x_{i}") for i in range(num_sets)]

    # Objective: minimize sum of costs
    objective = solver.Objective()
    for i in range(num_sets):
        objective.SetCoefficient(x[i], costs[i])
    objective.SetMinimization()

    # Constraints: each element must be covered at least once
    for elem in range(1, num_elements + 1):
        constraint = solver.Constraint(1.0, solver.infinity(), f"cover_{elem}")
        for set_idx, s in enumerate(sets):
            if elem in s:
                constraint.SetCoefficient(x[set_idx], 1.0)

    # Solve
    start_time = time.time()
    status = solver.Solve()
    solve_time = time.time() - start_time

    # Extract solution
    status_name = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status, "UNKNOWN")

    solution = [int(x[i].solution_value()) for i in range(num_sets)]

    # Get best bound for gap calculation
    best_bound = (
        solver.Objective().BestBound()
        if hasattr(solver.Objective(), "BestBound")
        else None
    )
    obj_value = (
        solver.Objective().Value()
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]
        else None
    )
    gap = None
    if obj_value and best_bound and obj_value > 0:
        gap = abs(obj_value - best_bound) / abs(obj_value)

    return {
        "status": status_name,
        "objective": obj_value,
        "solution": solution,
        "solve_time": solve_time,
        "gap": gap,
    }
