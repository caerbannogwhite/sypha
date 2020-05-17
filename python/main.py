
import numpy

from numpy.linalg import LinAlgError
from itertools import product
from pathlib import Path

from model_importer import *
from interior_point import *

def launch(path, instance):

    mat, rhs, obj = sc_model_reader(path)
    mat, rhs, obj = sc_to_standard_form(mat, rhs, obj)

    m, n = mat.shape

    try:
        x, y, s, iterations = mehrotra_linopt(mat, rhs, obj, 0.99995, 1000)
        print("{:10s} | {:10.4f} | {:10.4f} | {:3d}".format(instance, x[:n-m].dot(obj[:n-m]), y.dot(rhs), iterations))
    except LinAlgError:
        #print("LinAlgError on instance {}".format(i))
        pass

    del mat, rhs, obj, x, y, s


for i in range(50):
    path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scp_demo{:02d}.txt".format(i))
    launch(path, "scp_demo{:02d}".format(i))

#for i in range(10):
#    path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scp4{:d}.txt".format(i+1))
#    launch(path, "scp4{:d}".format(i+1))
#
#for p in product(["e", "f", "g", "h"], range(1,6)):
#    path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scpnr{}{}.txt".format(*p))
#    launch(path, "scpnr{}{}".format(*p))