ZERO_TOL = 1.E-20

def print_mat(mat):
    print("[", end="")
    for row in mat:
        print("[", end="")
        for e in row:
            v = 0.0 if abs(e) < ZERO_TOL else e
            print(f"{v:8.6f}, ", end="")
        print("],")
    print("]")


def print_vec(vec):
    print("[", end="")
    for e in vec:
        v = 0.0 if abs(e) < ZERO_TOL else e
        print(f"{v:8.6f}, ", end="")
    print("]")
