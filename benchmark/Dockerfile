# OR-Tools Benchmark for Set Cover Problem
FROM python:3.11-slim

WORKDIR /app

# Install OR-Tools
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy benchmark scripts
COPY scp_parser.py .
COPY ortools_solver.py .
COPY benchmark.py .

# Make benchmark script executable
RUN chmod +x benchmark.py

# Default: run benchmark on /data, output to /results
ENTRYPOINT ["python", "benchmark.py"]
CMD ["--data-dir", "/data", "--output", "/results/benchmark_results.csv"]
