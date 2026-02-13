# README

### TLDR

GPU-accelerated interior-point solver for Set Covering Problems (SCP).  
Currently a research / proof-of-concept implementation: **correct and reasonably robust, but not optimised or hardened for production use.**

It's very slow, for sure. 😎
But it works, for sure. 😎

### Model

`Model` is intended as an abstract interface capturing the structure of a mathematical program.  
Concrete instances are created, for example, when reading an LP file or an SCP file.

Planned/possible implementations:

- `ModelLP`
- `ModelMILP`

Key responsibilities:

- **Variables**: list of `Variable` objects (type, name, objective coefficient)
- **Constraints**: list of `Constraint` objects (row, name, sense, right-hand side)

### Node

A `Node` object stores all the information needed to derive its sub-model (via a crash / initialization procedure) from the original model.

### Solver

`solver_sparse_mehrotra`

At each interior-point iteration we solve the following linear system (twice, for affine and corrector steps):

```
      O | A' | I    x    -rc
      --|----|---   -    ---
      A | O  | O  * y  = -rb
      --|----|---   -    ---
      S | O  | X    s    -rxs
```

`A` is the model matrix in standard form, `I` is the \(n \times n\) identity
matrix, `S` is the \(n \times n\) diagonal matrix of `s`, and `X` is the \(n \times n\) diagonal matrix of `x`.  
The total number of non-zero elements in the block system is \(2 \cdot A.nnz + 3n\).

## SETTING UP

### Option A: Docker (recommended on Windows)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and enable WSL2
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU passthrough
3. Build and run:
   ```bash
   docker compose build
   docker compose run sypha
   ```
   Or with a specific data file:
   ```bash
   docker compose run sypha ./sypha --verbosity 100 --model SCP --input-file data/scp_demo00.txt
   ```

### Option B: Linux native build

- Install gsl library
  `sudo apt-get install libgsl-dev`

- Install boost
  `sudo apt-get install libboost-all-dev`

- Install CUDA
  for instance, to get cuda 11.0 just run
  `wget http://developer.download.nvidia.com/compute/cuda/11.0.1/local_installers/cuda_11.0.1_450.36.06_linux.run`
  `sudo sh cuda_11.0.1_450.36.06_linux.run`

(see Nvidia resources...)

- Build and run:
  `make`
  `./sypha --verbosity 100 --model SCP --input-file data/demo00.txt`

### Windows native build

The Makefile and source use POSIX APIs (`gettimeofday`, `unistd.h`) and are designed for Linux. To build on Windows, use the Docker option above, or WSL2 with Ubuntu to run the Linux build natively.

## BENCHMARKING WITH OR-TOOLS

A Python-based benchmark suite using [Google OR-Tools](https://developers.google.com/optimization) is available in the `benchmark/` directory.

### Run Benchmarks

```bash
cd benchmark

# Linear relaxation only (fast)
docker compose run --rm benchmark

# With integer solutions (slower, 300s time limit per instance)
docker compose run --rm benchmark-with-integer
```

Results are saved to `benchmark/results/benchmark_results.csv`.

See `benchmark/README.md` for deployment on AWS and advanced usage.
