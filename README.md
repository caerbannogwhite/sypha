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

## SETTING UP ##

* Install gsl library
`sudo apt-get install libgsl-dev`

* Install boost
`sudo apt-get install libboost-all-dev`

* Install CUDA
for instance, to get cuda 11.0 just run
`wget http://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda_11.0.1_450.36.06_linux.run`
`sudo sh cuda_11.0.1_450.36.06_linux.run`

(see Nvidia resources...)
