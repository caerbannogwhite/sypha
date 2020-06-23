import re
import os

from statistics import mean, stdev
from argparse import ArgumentParser

ACCEPT_TOL = 1.E-4
BASE_DIR = "/home/macs/coding/optimization/sypha"

SCP4_STRINGS = [f"scp4{i+1:d}" for i in range(10)]
SCP5_STRINGS = [f"scp5{i+1:d}" for i in range(10)]
SCPNRE_STRINGS = [f"scpnre{i+1:d}" for i in range(5)]
SCPNRF_STRINGS = [f"scpnrf{i+1:d}" for i in range(5)]
SCPNRG_STRINGS = [f"scpnrg{i+1:d}" for i in range(5)]
SCPNRH_STRINGS = [f"scpnrh{i+1:d}" for i in range(5)]


SCP4_SOLUTIONS = [
    429.00000000000000000000,
    512.00000000000000000000,
    516.00000000000000000000,
    494.00000000000000000000,
    512.00000000000000000000,
    557.25000000000000000000,
    430.00000000000000000000,
    488.66666666666662877105,
    638.53846153846154720668,
    513.50000000000000000000
]

SCP5_SOLUTIONS = [
    251.22499999999996589395,
    299.76111111111111995342,
    226.00000000000000000000,
    240.50000000000000000000,
    211.00000000000000000000,
    212.50000000000000000000,
    291.77777777777782830526,
    287.00000000000000000000,
    279.00000000000000000000,
    265.00000000000000000000
]

SCPNRE_SOLUTIONS = [
    21.37941620724624769423,
    22.36004487360803949514,
    20.48614223624254293554,
    21.35271525505882195262,
    21.32192095818086841064
]

SCPNRF_SOLUTIONS = [
    8.79526382275696150259,
    9.99361516000088556666,
    9.49237692915252395665,
    8.47119009228243236009,
    7.83552724858639937366
]

SCPNRG_SOLUTIONS = [
    159.88624078126431982128,
    142.07332051900436908909,
    148.26913540494277299331,
    148.94652093714017837556,
    148.23146550380926100843
]

SCPNRH_SOLUTIONS = [
    48.12455464179099351441,
    48.63762489585338499865,
    45.19746213904625165014,
    44.04210816470045131155,
    42.37035886823193209239
]


def exec_sypha(inst, solution, repeat):
    data = dict()
    log_file_name = f"{inst}_test.log"

    pass_cnt = 0
    fail_cnt = 0
    iterations = []
    start_sol_times = []
    pre_sol_times = []
    solver_times = []
    total_times = []

    for _ in range(repeat):
        os.system(f"{BASE_DIR}/sypha --model SCP --verbosity 1 "
                  f"--input-file {BASE_DIR}/data/{inst}.txt > {log_file_name}")
    
        file_handler = open(log_file_name, "r")
        accept_prim = False
        accept_dual = False
        for row in file_handler:
            if "PRIMAL" in row:
                p, prim = row.split(":")
                accept_prim = abs(float(prim) - solution) < ACCEPT_TOL
            elif "DUAL" in row:
                p, dual = row.split(":")
                accept_dual = abs(float(dual) - solution) < ACCEPT_TOL
            elif "ITERATIONS" in row:
                p, val = row.split(":")
                iterations.append(int(val))
            elif "TIME START SOL" in row:
                p, val = row.split(":")
                start_sol_times.append(float(val) / 1000)
            elif "TIME PRE SOL" in row:
                p, val = row.split(":")
                pre_sol_times.append(float(val) / 1000)
            elif "TIME SOLVER" in row:
                p, val = row.split(":")
                solver_times.append(float(val) / 1000)
            elif "TIME TOTAL" in row:
                p, val = row.split(":")
                total_times.append(float(val) / 1000)

        if accept_prim and accept_dual:
            pass_cnt += 1
        else:
            fail_cnt += 1

        file_handler.close()
    
    data["inst"] = inst
    data["pass"] = pass_cnt
    data["fail"] = fail_cnt
    data["repeat"] = repeat
    data["iterations_mean"] = mean(iterations)
    data["start_sol_time_mean"] = mean(start_sol_times)
    data["start_sol_time_std"] = stdev(start_sol_times)
    data["pre_sol_time_mean"] = mean(pre_sol_times)
    data["pre_sol_time_std"] = stdev(pre_sol_times)
    data["solver_time_mean"] = mean(solver_times)
    data["solver_time_std"] = stdev(solver_times)
    data["total_time_mean"] = mean(total_times)
    data["total_time_std"] = stdev(total_times)

    return data



def launch_tests(match, repeat):
    log_file = open("test.log", "w")

    # header
    log_file.write(",".join([
        "INST",
        "ITER MEAN",
        "SOLVER TIME MEAN (s)",
        "SOLVER TIME STD",
        "TOTAL TIME MEAN (s)",
        "TOTAL TIME STD",
        "REP",
        "PASS",
        "FAIL",
    ]) + "\n")

    for i, instance in enumerate(SCP4_STRINGS):
        if re.match(match, instance):
            res = exec_sypha(instance, SCP4_SOLUTIONS[i], repeat)
            log_file.write(",".join(map(str, [
                res["inst"],
                res["iterations_mean"],
                res["solver_time_mean"],
                res["solver_time_std"],
                res["total_time_mean"],
                res["total_time_std"],
                res["repeat"],
                res["pass"],
                res["fail"]
            ])) + "\n")

    for i, instance in enumerate(SCP5_STRINGS):
        if re.match(match, instance):
            res = exec_sypha(instance, SCP5_SOLUTIONS[i], repeat)
            log_file.write(",".join(map(str, [
                res["inst"],
                res["iterations_mean"],
                res["solver_time_mean"],
                res["solver_time_std"],
                res["total_time_mean"],
                res["total_time_std"],
                res["repeat"],
                res["pass"],
                res["fail"]
            ])) + "\n")

    log_file.close()


if __name__ == "__main__":
    # parser = ArgumentParser("Sypha unit tests.")
    # parser.add_argument("--test-match", type=str, help="")
    # parser.add_argument("--repeat", type=int, help="Number of repetition")

    launch_tests("scp[4-5]*", 3)
