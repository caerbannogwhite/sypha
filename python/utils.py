def print_mat(mat):
    print("[", end="")
    for row in mat:
        print("[", end="")
        for e in row:
            print(f"{e:8.6f}, ", end="")
        print("],")
    print("]")


def print_vec(vec):
    print("[", end="")
    for e in vec:
        print(f"{e:8.6f}, ", end="")
    print("]")
