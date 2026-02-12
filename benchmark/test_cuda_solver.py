#!/usr/bin/env python3
"""
Test CUDA solver against OR-Tools benchmark results.
Runs Sypha CUDA solver and compares with OR-Tools LP objectives.
"""

import subprocess
import re
import csv
from pathlib import Path
import pandas as pd


def run_cuda_solver(instance_path):
    """
    Run CUDA solver on an instance and extract primal/dual objectives.

    Returns:
        dict: {
            'primal': float or None,
            'dual': float or None,
            'iterations': int or None,
            'time_solver': float or None,
            'status': str (SUCCESS or ERROR)
        }
    """
    try:
        # Run docker command
        cmd = [
            "docker",
            "compose",
            "run",
            "--rm",
            "sypha",
            "./sypha",
            "--verbosity",
            "100",
            "--model",
            "SCP",
            "--input-file",
            f"data/{instance_path.name}",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(instance_path.parent.parent),  # Go to sypha root
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = result.stdout + result.stderr

        # Parse output for PRIMAL, DUAL, ITERATIONS, TIME SOLVER
        primal_match = re.search(
            r"PRIMAL:\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", output
        )
        dual_match = re.search(
            r"DUAL:\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", output
        )
        iter_match = re.search(r"ITERATIONS:\s+(\d+)", output)
        time_match = re.search(r"TIME SOLVER:\s+([-+]?[0-9]*\.?[0-9]+)", output)

        return {
            "primal": float(primal_match.group(1)) if primal_match else None,
            "dual": float(dual_match.group(1)) if dual_match else None,
            "iterations": int(iter_match.group(1)) if iter_match else None,
            "time_solver": float(time_match.group(1)) if time_match else None,
            "status": "SUCCESS" if primal_match and dual_match else "ERROR",
            "output": output,
        }

    except subprocess.TimeoutExpired:
        return {
            "primal": None,
            "dual": None,
            "iterations": None,
            "time_solver": None,
            "status": "TIMEOUT",
            "output": "Timeout after 60s",
        }
    except Exception as e:
        return {
            "primal": None,
            "dual": None,
            "iterations": None,
            "time_solver": None,
            "status": "ERROR",
            "output": str(e),
        }


def compare_results(
    ortools_csv, data_dir, num_instances=20, output_csv="cuda_comparison.csv"
):
    """
    Compare CUDA solver results with OR-Tools benchmark.

    Args:
        ortools_csv: path to OR-Tools benchmark CSV
        data_dir: directory with SCP instance files
        num_instances: number of instances to test
        output_csv: output comparison CSV
    """
    # Read OR-Tools results
    df_ortools = pd.read_csv(ortools_csv)

    # Take first N instances
    df_ortools = df_ortools.head(num_instances)

    results = []

    for idx, row in df_ortools.iterrows():
        instance_name = row["instance"]
        ortools_lp_obj = row["lp_objective"]
        ortools_status = row["lp_status"]

        print(f"\n[{idx+1}/{num_instances}] Testing {instance_name}...", flush=True)
        print(f"  OR-Tools LP: {ortools_lp_obj} ({ortools_status})")

        # Run CUDA solver
        instance_path = Path(data_dir) / instance_name
        cuda_result = run_cuda_solver(instance_path)

        print(f"  CUDA Primal: {cuda_result['primal']}")
        print(f"  CUDA Dual:   {cuda_result['dual']}")

        # Compare
        match = False
        diff_primal = None
        diff_dual = None

        if cuda_result["status"] == "SUCCESS" and ortools_status == "OPTIMAL":
            if ortools_lp_obj is not None and cuda_result["primal"] is not None:
                diff_primal = abs(cuda_result["primal"] - ortools_lp_obj)
                diff_dual = (
                    abs(cuda_result["dual"] - ortools_lp_obj)
                    if cuda_result["dual"]
                    else None
                )

                # Consider match if within 0.1% or 0.01 absolute
                tol_rel = 0.001
                tol_abs = 0.01
                match = (
                    diff_primal < tol_abs
                    or diff_primal / max(abs(ortools_lp_obj), 1e-10) < tol_rel
                )

                if match:
                    print(f"  ✓ MATCH (diff: {diff_primal:.6f})")
                else:
                    print(f"  ✗ MISMATCH (diff: {diff_primal:.6f})")
        elif ortools_status == "INFEASIBLE":
            print(f"  (OR-Tools says INFEASIBLE, skipping comparison)")
            match = None  # Can't compare infeasible instances

        results.append(
            {
                "instance": instance_name,
                "ortools_status": ortools_status,
                "ortools_lp_objective": ortools_lp_obj,
                "cuda_status": cuda_result["status"],
                "cuda_primal": cuda_result["primal"],
                "cuda_dual": cuda_result["dual"],
                "cuda_iterations": cuda_result["iterations"],
                "cuda_time_solver": cuda_result["time_solver"],
                "diff_primal": diff_primal,
                "diff_dual": diff_dual,
                "match": match,
            }
        )

    # Write comparison CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\n\nComparison results written to {output_csv}")

    # Summary
    if "match" in df_results.columns:
        matches = df_results["match"].sum()
        total_comparable = df_results["match"].notna().sum()
        print(
            f"\nSummary: {matches}/{total_comparable} instances match OR-Tools results"
        )

        # Show mismatches
        mismatches = df_results[df_results["match"] == False]
        if not mismatches.empty:
            print("\nMismatches:")
            for _, row in mismatches.iterrows():
                print(
                    f"  {row['instance']}: OR-Tools={row['ortools_lp_objective']:.6f}, "
                    f"CUDA={row['cuda_primal']:.6f}, diff={row['diff_primal']:.6f}"
                )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test CUDA solver against OR-Tools")
    parser.add_argument(
        "--ortools-csv",
        default="results/benchmark_results_with_ip.csv",
        help="OR-Tools benchmark CSV",
    )
    parser.add_argument(
        "--data-dir", default="../data", help="Directory with SCP instances"
    )
    parser.add_argument(
        "--num-instances", type=int, default=20, help="Number of instances to test"
    )
    parser.add_argument(
        "--output", default="results/cuda_comparison.csv", help="Output comparison CSV"
    )

    args = parser.parse_args()

    compare_results(args.ortools_csv, args.data_dir, args.num_instances, args.output)


if __name__ == "__main__":
    main()
