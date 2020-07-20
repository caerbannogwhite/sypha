
import argparse
import numpy
import re
import os

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
                    if solver == "dense" or solver == "dense_test":
                        mat, rhs, obj = sc_dense_model_reader(BASE_DIR + entry)
                        mat, rhs, obj = sc_dense_to_standard_form(mat, rhs, obj)

                        if solver == "dense":
                            x, y, s, iterations = mehrotra_linopt_dense(mat, rhs, obj)
                        else:
                            x, y, s, iterations = mehrotra_linopt_dense_test(mat, rhs, obj)                        

                    elif  solver == "sparse":
                        mat, rhs, obj = sc_sparse_model_reader(path)
                        mat, rhs, obj = sc_sparse_to_standard_form(mat, rhs, obj)

                        x, y, s, iterations = mehrotra_linopt_sparse(mat, rhs, obj)

                    else:
                        print("Solver not found")

                    m, n = mat.shape

                    upp = x[:n-m].dot(obj[:n-m])
                    low = y.dot(rhs)
                    if numpy.isclose(upp, low, 1E-6):
                        flag = True

                    print(f"{name:15s} ({m:5d},{n:5d}) | {upp:16.6f} | {low:16.6f} | {iterations:4d}")
                    del x, y, s

                except numpy.linalg.LinAlgError:
                    print("lin alg error")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--instance", dest="instance", type=str)
    parser.add_argument("-s", "--solver", dest="solver", type=str)

    args = parser.parse_args()
    launch(args.instance, args.solver)


