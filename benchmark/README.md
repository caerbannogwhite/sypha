# OR-Tools SCP Benchmark

This benchmark suite solves Set Cover Problem (SCP) instances using [Google OR-Tools](https://developers.google.com/optimization).

## Features

- **Linear Relaxation**: Solves LP relaxation using GLOP (Google's LP solver)
- **Integer Programming**: Solves integer version using SCIP
- **Primal & Dual Solutions**: Extracts both primal variables and constraint duals
- **CSV Output**: Results exported to CSV for analysis
- **Docker**: Containerized for easy deployment on AWS or other cloud platforms

## Quick Start

### Run Linear Relaxation Only (Fast)

```bash
cd benchmark
docker compose build
docker compose run --rm benchmark
```

Results will be in `benchmark/results/benchmark_results.csv`.

### Run with Integer Solutions (Slower)

```bash
docker compose run --rm benchmark-with-integer
```

Results will be in `benchmark/results/benchmark_results_with_ip.csv`.

### Custom Options

```bash
# Custom time limit for IP solver (default 300s)
docker compose run --rm benchmark python benchmark.py \
    --data-dir /data \
    --output /results/custom_results.csv \
    --solve-integer \
    --time-limit 600
```

## Output Format

CSV columns:
- `instance`: filename
- `num_sets`, `num_elements`: problem size
- `lp_status`, `lp_objective`, `lp_solve_time`: LP relaxation results
- `ip_status`, `ip_objective`, `ip_solve_time`, `ip_gap`: IP results (if `--solve-integer`)
- `error`: any error messages

## Deployment on AWS

### Option 1: EC2 Instance

```bash
# SSH to EC2 instance
ssh -i your-key.pem ubuntu@your-instance

# Clone repo and run
git clone your-repo
cd sypha/benchmark
docker compose run --rm benchmark-with-integer
```

### Option 2: AWS Batch

1. Build and push Docker image:

```bash
cd benchmark
docker build -t your-ecr-repo/scp-benchmark:latest .
docker push your-ecr-repo/scp-benchmark:latest
```

2. Create AWS Batch job definition using the image
3. Submit job with environment variables or command overrides

### Option 3: ECS/Fargate

Use the Docker image with ECS task definition and mount S3 for results.

## Advanced Usage

### Run a Single Instance

```python
from scp_parser import parse_scp_file
from ortools_solver import solve_scp_linear_relaxation, solve_scp_integer

problem = parse_scp_file('/data/scp_demo00.txt')
lp_result = solve_scp_linear_relaxation(problem)
ip_result = solve_scp_integer(problem)

print(f"LP objective: {lp_result['objective']}")
print(f"IP objective: {ip_result['objective']}")
```

## Requirements

- Docker
- Python 3.11+ (if running locally)
- OR-Tools 9.11+

## License

Same as parent project (Sypha).
