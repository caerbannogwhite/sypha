# README #

### Model ###

Model should be an abstract class to collect the features of a mathematical model. A Model instance
should be returned, for instance, when reading an LP file or and SCP file.
Possible implementations:
* ModelLP
* ModelMILP

Main features:
* Variables: a list of Variable objects (type, name, objective)
* Constrains: a list of Constraint objects (row, name, sense, rhs)

### Node ###

A Node object maintains the information to get its sub-model representation (crash procedure) from
the original model.

### Solver ###

`solver_sparse_merhrotra`
At each iteration we solve this linear system twice:

```
      O | A' | I    x    -rc
      --|----|---   -    ---
      A | O  | O  * y  = -rb
      --|----|---   -    ---
      S | O  | X    s    -rxs
```

`A` is the model matrix (in *standard form*), `I` is the *n * n* identity
matrix, `S` is the *n * n* `s` diagonal matrix, `X` is the *n * n* `x` diagonal matrix.
Total number of non-zero elements is *`A.nnz` * 2 + n * 3*.

## ISSUES ##

* SYPHA-1: `sypha_node_sparse` - change X, S matrix update phase on `solver_sparse_merhrotra` using cublas copy
* SYPHA-2: `sypha_node_sparse` - update the procedure to find `alphaMaxPrim` and `alphaMaxDual` on `solver_sparse_merhrotra` using device kernels
* SYPHA-3: `sypha_node_sparse` - improve initialization of big matrix A
* SYPHA-4: `sypha_node_sparse` - develop the Merhrotra procedure not using the big matrix A and test performances
* SYPHA-5: check file path before reading it