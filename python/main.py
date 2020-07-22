
import argparse
import numpy
import re
import os
import time

from numpy.linalg import LinAlgError
from itertools import product
from pathlib import Path

from model_importer import *
from interior_point import *


BASE_DIR = "/home/macs/coding/optimization/sypha/data/"

def launch(instance, solver):

    for entry in os.listdir(BASE_DIR):
        
        if os.path.isfile(BASE_DIR + entry):
            name, ext = entry.split(".")
            if re.match(instance, name):

                try:
                    if solver == "dense":
                        mat, rhs, obj = sc_dense_model_reader(BASE_DIR + entry)
                        mat, rhs, obj = sc_dense_to_standard_form(mat, rhs, obj)

                        time_start = time.perf_counter()
                        x, y, s, iterations = mehrotra_linopt_dense(mat, rhs, obj)
                        time_delta = time.perf_counter() - time_start

                    elif  solver == "dense_test":
                        mat, rhs, obj = sc_dense_model_reader(BASE_DIR + entry)
                        mat, rhs, obj = sc_dense_to_standard_form(mat, rhs, obj)

                        time_start = time.perf_counter()
                        x, y, s, iterations = mehrotra_linopt_dense_test(mat, rhs, obj)
                        time_delta = time.perf_counter() - time_start

                    elif solver == "dense_test_2":
                        mat, rhs, obj = sc_dense_model_reader(BASE_DIR + entry)
                        mat, rhs, obj = sc_dense_to_standard_form(
                            mat, rhs, obj)

                        time_start = time.perf_counter()
                        x, y, s, iterations = mehrotra_linopt_dense_test_2(
                            mat, rhs, obj)
                        time_delta = time.perf_counter() - time_start

                    elif solver == "alm_test":
                        mat, rhs, obj = sc_dense_model_reader(BASE_DIR + entry)
                        mat, rhs, obj = sc_dense_to_standard_form(mat, rhs, obj)
                        
                        m, n = mat.shape
                        lb = numpy.zeros(n)
                        ub = numpy.ones(n)
                        ub[n-m:] = numpy.inf

                        time_start = time.perf_counter()
                        x, y, s, iterations = alm_test(mat, rhs, obj, lb, ub)
                        time_delta = time.perf_counter() - time_start

                    elif  solver == "sparse":
                        mat, rhs, obj = sc_sparse_model_reader(BASE_DIR + entry)
                        mat, rhs, obj = sc_sparse_to_standard_form(mat, rhs, obj)

                        time_start = time.perf_counter()
                        x, y, s, iterations = mehrotra_linopt_sparse(mat, rhs, obj)
                        time_delta = time.perf_counter() - time_start

                    else:
                        print("Solver not found")

                    m, n = mat.shape

                    upp = x[:n-m].dot(obj[:n-m])
                    low = y.dot(rhs)
                    if numpy.isclose(upp, low, 1E-6):
                        flag = True

                    print(f"{name:15s} ({m:5d},{n:5d}) | {upp:16.6f} | {low:16.6f} | {iterations:4d} | {time_delta:10.4f}")
                    del x, y, s

                except numpy.linalg.LinAlgError:
                    print("lin alg error")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--instance", dest="instance", type=str)
    parser.add_argument("-s", "--solver", dest="solver", type=str)

    args = parser.parse_args()
    launch(args.instance, args.solver)


