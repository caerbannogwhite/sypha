
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
    flag = False

    try:
        x, y, s, iterations = mehrotra_linopt(mat, rhs, obj)
        upp = x[:n-m].dot(obj[:n-m])
        low = y.dot(rhs)
        if numpy.isclose(upp, low, 1E-6):
            flag = True

        print("{:10s} | {:16.6f} | {:16.6f} | {:4d}".format(instance, upp, low, iterations))
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
#    if launch(path, "scp_demo{:02d}".format(i)):
#        counter_ok += 1

for i in range(10):
    path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scp4{:d}.txt".format(i+1))
    counter += 1
    if launch(path, "scp4{:d}".format(i+1)):
        counter_ok += 1

#for p in product(["e",], range(1,6)):
#    path = Path("C:\\Users\\IP 520S-14IKB 96IX\\coding\\sypha\\data\\scpnr{}{}.txt".format(*p))
#    launch(path, "scpnr{}{}".format(*p))

print(counter, counter_ok)