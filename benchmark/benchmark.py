#!/usr/bin/env python3
"""
Benchmark script for Set Cover Problem using Google OR-Tools.
Runs on all .txt files in the data/ directory.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import traceback

from scp_parser import parse_scp_file
from ortools_solver import solve_scp_linear_relaxation, solve_scp_integer


def run_benchmark(data_dir, output_csv, solve_integer=False, time_limit=300):
    """
    Run benchmark on all SCP files in data_dir.

    Args:
        data_dir: path to directory with .txt files
        output_csv: output CSV file path
        solve_integer: if True, also solve integer version
        time_limit: time limit for integer solver (seconds)
    """
    data_path = Path(data_dir)

    # Get all .txt files
    instance_files = sorted(data_path.glob("*.txt"))

    if not instance_files:
        print(f"No .txt files found in {data_dir}")
        return

    print(f"Found {len(instance_files)} instance files")

    # Prepare CSV
    fieldnames = [
        "instance",
        "num_sets",
        "num_elements",
        "lp_status",
        "lp_objective",
        "lp_solve_time",
        "ip_status",
        "ip_objective",
        "ip_solve_time",
        "ip_gap",
        "error",
    ]

    results = []

    for idx, filepath in enumerate(instance_files, 1):
        instance_name = filepath.name
        print(
            f"\n[{idx}/{len(instance_files)}] Processing {instance_name}...", flush=True
        )

        result = {
            "instance": instance_name,
            "num_sets": None,
            "num_elements": None,
            "lp_status": None,
            "lp_objective": None,
            "lp_solve_time": None,
            "ip_status": None,
            "ip_objective": None,
            "ip_solve_time": None,
            "ip_gap": None,
            "error": None,
        }

        try:
            # Parse instance
            problem_data = parse_scp_file(filepath)
            result["num_sets"] = problem_data["num_sets"]
            result["num_elements"] = problem_data["num_elements"]

            print(
                f"  Problem size: {problem_data['num_sets']} sets, {problem_data['num_elements']} elements"
            )

            # Solve linear relaxation
            print("  Solving linear relaxation...", end="", flush=True)
            lp_result = solve_scp_linear_relaxation(problem_data)
            result["lp_status"] = lp_result["status"]
            result["lp_objective"] = lp_result["objective"]
            result["lp_solve_time"] = lp_result["solve_time"]
            print(
                f" {lp_result['status']} in {lp_result['solve_time']:.3f}s, obj={lp_result['objective']}"
            )

            # Solve integer version if requested
            if solve_integer:
                print(
                    f"  Solving integer program (limit {time_limit}s)...",
                    end="",
                    flush=True,
                )
                ip_result = solve_scp_integer(problem_data, time_limit_sec=time_limit)
                result["ip_status"] = ip_result["status"]
                result["ip_objective"] = ip_result["objective"]
                result["ip_solve_time"] = ip_result["solve_time"]
                result["ip_gap"] = ip_result["gap"]
                gap_str = f", gap={ip_result['gap']:.2%}" if ip_result["gap"] else ""
                print(
                    f" {ip_result['status']} in {ip_result['solve_time']:.3f}s, obj={ip_result['objective']}{gap_str}"
                )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result["error"] = error_msg
            print(f"  ERROR: {error_msg}")
            traceback.print_exc()

        results.append(result)

    # Write CSV
    print(f"\n\nWriting results to {output_csv}...")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmark complete. Results saved to {output_csv}")

    # Print summary
    lp_optimal = sum(1 for r in results if r["lp_status"] == "OPTIMAL")
    if solve_integer:
        ip_optimal = sum(1 for r in results if r["ip_status"] == "OPTIMAL")
        ip_feasible = sum(1 for r in results if r["ip_status"] == "FEASIBLE")
        print(f"\nSummary:")
        print(f"  LP:  {lp_optimal}/{len(results)} optimal")
        print(
            f"  IP:  {ip_optimal}/{len(results)} optimal, {ip_feasible}/{len(results)} feasible"
        )
    else:
        print(f"\nSummary: {lp_optimal}/{len(results)} LP optimal")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SCP instances with OR-Tools"
    )
    parser.add_argument(
        "--data-dir", default="/data", help="Directory with SCP .txt files"
    )
    parser.add_argument(
        "--output", default="/results/benchmark_results.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--solve-integer", action="store_true", help="Also solve integer version"
    )
    parser.add_argument(
        "--time-limit", type=int, default=300, help="Time limit for IP solver (seconds)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_benchmark(args.data_dir, args.output, args.solve_integer, args.time_limit)


if __name__ == "__main__":
    main()
