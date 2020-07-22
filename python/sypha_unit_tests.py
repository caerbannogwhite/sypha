#! /usr/bin/python3

import re
import os

from datetime import datetime
from statistics import mean, stdev
from argparse import ArgumentParser

ACCEPT_TOL = 1.E-4
BASE_DIR = "/home/macs/coding/optimization/sypha"
#BASE_DIR = "/home/ubuntu/sypha"

class Entry(object):

    HEADER = [
        "INST",
        "ITER MEAN",
        "SOLVER TIME MEAN (s)",
        "SOLVER TIME STD",
        "TOTAL TIME MEAN (s)",
        "TOTAL TIME STD",
        "REP",
        "PASS",
        "FAIL",
    ]

    def __init__(self, instance, repeat):
        self.instance = instance
        self.repeat = repeat
        self.passed = 0
        self.failed = 0
        self.iterations_vals = []
        self.start_sol_time_vals = []
        self.pre_sol_time_vals = []
        self.solver_time_vals = []
        self.total_time_vals = []

    @staticmethod
    def get_csv_header(sep=","):
        return sep.join(Entry.HEADER) + "\n"
        
    def to_csv(self, sep=","):
        vals = [
            self.instance,
            mean(self.iterations_vals),
            # mean(self.start_sol_time_vals),
            # 0.0 if len(self.start_sol_time_vals) < 2 else stdev(self.start_sol_time_vals),
            # mean(self.pre_sol_time_vals),
            # 0.0 if len(self.pre_sol_time_vals) < 2 else stdev(self.pre_sol_time_vals),
            mean(self.solver_time_vals),
            0.0 if len(self.solver_time_vals) < 2 else stdev(self.solver_time_vals),
            mean(self.total_time_vals),
            0.0 if len(self.total_time_vals) < 2 else stdev(self.total_time_vals),
            self.repeat,
            self.passed,
            self.failed,
        ]
        return sep.join(map(str, vals)) + "\n"

    def __repr__(self):
        return ""


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
    
    entry = Entry(inst, repeat)

    for i in range(repeat):
        log_file_name = f"{inst}_test_{i}_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
        os.system(f"{BASE_DIR}/sypha --model SCP --verbosity 1 --tol-mu 1.E-8 "
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
                entry.iterations_vals.append(int(val))
            elif "TIME START SOL" in row:
                p, val = row.split(":")
                entry.start_sol_time_vals.append(float(val) / 1000)
            elif "TIME PRE SOL" in row:
                p, val = row.split(":")
                entry.pre_sol_time_vals.append(float(val) / 1000)
            elif "TIME SOLVER" in row:
                p, val = row.split(":")
                entry.solver_time_vals.append(float(val) / 1000)
            elif "TIME TOTAL" in row:
                p, val = row.split(":")
                entry.total_time_vals.append(float(val) / 1000)

        if accept_prim and accept_dual:
            entry.passed += 1
        else:
            entry.failed += 1

        file_handler.close()

    return entry


def launch_tests(match, repeat):
    log_file = open(f"result_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "w")

    # header
    log_file.write(Entry.get_csv_header())

    for i, instance in enumerate(SCP4_STRINGS):
        if re.match(match, instance):
            entry = exec_sypha(instance, SCP4_SOLUTIONS[i], repeat)
            log_file.write(entry.to_csv())

    for i, instance in enumerate(SCP5_STRINGS):
        if re.match(match, instance):
            entry = exec_sypha(instance, SCP5_SOLUTIONS[i], repeat)
            log_file.write(entry.to_csv())

    for i, instance in enumerate(SCPNRE_STRINGS):
        if re.match(match, instance):
            entry = exec_sypha(instance, SCPNRE_SOLUTIONS[i], repeat)
            log_file.write(entry.to_csv())

    for i, instance in enumerate(SCPNRF_STRINGS):
        if re.match(match, instance):
            entry = exec_sypha(instance, SCPNRF_SOLUTIONS[i], repeat)
            log_file.write(entry.to_csv())

    for i, instance in enumerate(SCPNRG_STRINGS):
        if re.match(match, instance):
            entry = exec_sypha(instance, SCPNRG_SOLUTIONS[i], repeat)
            log_file.write(entry.to_csv())

    for i, instance in enumerate(SCPNRH_STRINGS):
        if re.match(match, instance):
            entry = exec_sypha(instance, SCPNRH_SOLUTIONS[i], repeat)
            log_file.write(entry.to_csv())

    log_file.close()


if __name__ == "__main__":
    parser = ArgumentParser("Sypha unit tests.")
    parser.add_argument("-m", "--test-match", dest="match", type=str, help="")
    parser.add_argument("-r", "--repeat", dest="repeat", type=int, help="Number of repetition")

    args = parser.parse_args()
    launch_tests(args.match, args.repeat)
