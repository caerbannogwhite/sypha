
import numpy

from numpy.linalg import LinAlgError
from itertools import product
from pathlib import Path

from model_importer import *
from interior_point import *


def launch(path, instance, sparse=False):

    if sparse:
        mat, rhs, obj = sc_sparse_model_reader(path)
        mat, rhs, obj = sc_sparse_to_standard_form(mat, rhs, obj)
    else:
        mat, rhs, obj = sc_dense_model_reader(path)
        mat, rhs, obj = sc_dense_to_standard_form(mat, rhs, obj)

    m, n = mat.shape
    flag = False

    try:
        if sparse:
            x, y, s, iterations = mehrotra_linopt_sparse(mat, rhs, obj)
        else:
            x, y, s, iterations = mehrotra_linopt_dense(mat, rhs, obj)

        upp = x[:n-m].dot(obj[:n-m])
        low = y.dot(rhs)
        if numpy.isclose(upp, low, 1E-6):
            flag = True

        print(f"{instance:10s} ({m:3d},{n:3d}) | {upp:16.6f} | {low:16.6f} | {iterations:4d}")
        del x, y, s
    except LinAlgError:
        #print("LinAlgError on instance {}".format(i))
        pass

    del mat, rhs, obj
    return flag

#path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\ex_balas1.txt")
#launch(path, "ex_balas1")

counter = 0
counter_ok = 0
#for i in range(50):
#    path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scp_demo{:02d}.txt".format(i))
#    counter += 1
#    if launch(path, f"scp_demo{i:02d}", sparse=True):
#        counter_ok += 1

for i in range(10):
    #path = Path(f"C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scp4{i+1}.txt")
    path = Path(f"/home/macs/coding/optimization/sypha/data/scp4{i+1}.txt")
    counter += 1
    if launch(path, "scp4{:d}".format(i+1)):
        counter_ok += 1

#for i in range(10):
#    path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scp5{:d}.txt".format(i+1))
#    counter += 1
#    if launch(path, f"scp5{i+1:d}", sparse=True):
#        counter_ok += 1

#for p in product(["e", "f",], range(1,6)):
#    path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scpnr{}{}.txt".format(*p))
#    launch(path, "scpnr{}{}".format(*p), sparse=True)

print(counter, counter_ok)
