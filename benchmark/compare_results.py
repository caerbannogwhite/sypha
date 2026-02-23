#!/usr/bin/env python3
"""Compare Sypha and OR-Tools benchmark results."""

import csv
import sys
from pathlib import Path


def load_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["instance"]] = row
    return rows


def main():
    results_dir = Path(__file__).parent / "results"
    ortools_csv = results_dir / "benchmark_results_with_ip.csv"
    sypha_csv = results_dir / "sypha_results.csv"

    if not ortools_csv.exists():
        print(f"OR-Tools results not found: {ortools_csv}")
        sys.exit(1)
    if not sypha_csv.exists():
        print(f"Sypha results not found: {sypha_csv}")
        sys.exit(1)

    ortools = load_csv(ortools_csv)
    sypha = load_csv(sypha_csv)

    # Filter to instances present in both
    common = sorted(set(ortools.keys()) & set(sypha.keys()))

    if not common:
        print("No common instances found!")
        sys.exit(1)

    # Print comparison table
    hdr = f"{'Instance':<14} {'OR-LP':>10} {'OR-IP':>8} {'OR-IP-t':>8} {'Sypha-Inc':>10} {'Sypha-Gap':>10} {'Sypha-t':>8} {'Sypha-St':>10} {'IP-Match':>9}"
    print(hdr)
    print("-" * len(hdr))

    matches = 0
    total = 0
    total_ortools_time = 0.0
    total_sypha_time = 0.0
    sypha_better = 0
    ortools_better = 0

    for inst in common:
        ot = ortools[inst]
        sy = sypha[inst]

        or_lp = ot.get("lp_objective", "")
        or_ip = ot.get("ip_objective", "")
        or_ip_time = ot.get("ip_solve_time", "")
        or_ip_status = ot.get("ip_status", "")

        sy_inc = sy.get("incumbent", "")
        sy_gap = sy.get("mip_gap_pct", "")
        sy_time = sy.get("time_total_s", "")
        sy_status = sy.get("status", "")

        # Format values
        or_lp_f = f"{float(or_lp):.2f}" if or_lp else "n/a"
        or_ip_f = f"{float(or_ip):.0f}" if or_ip else "n/a"
        or_ip_t = f"{float(or_ip_time):.1f}s" if or_ip_time else "n/a"
        sy_inc_f = f"{float(sy_inc):.0f}" if sy_inc else "n/a"
        sy_gap_f = f"{float(sy_gap):.2f}%" if sy_gap else "n/a"
        sy_time_f = f"{float(sy_time):.1f}s" if sy_time else "n/a"

        # Compare IP objectives
        ip_match = ""
        if or_ip and sy_inc:
            try:
                or_val = float(or_ip)
                sy_val = float(sy_inc)
                total += 1
                if or_ip_time:
                    total_ortools_time += float(or_ip_time)
                if sy_time:
                    total_sypha_time += float(sy_time)

                if abs(or_val - sy_val) < 0.5:
                    ip_match = "MATCH"
                    matches += 1
                elif sy_val < or_val:
                    ip_match = "SYPHA+"
                    sypha_better += 1
                else:
                    ip_match = "ORTOOLS+"
                    ortools_better += 1
            except ValueError:
                ip_match = "ERR"
        elif sy_inc:
            ip_match = "no-ref"
        else:
            ip_match = "no-sol"

        print(f"{inst:<14} {or_lp_f:>10} {or_ip_f:>8} {or_ip_t:>8} {sy_inc_f:>10} {sy_gap_f:>10} {sy_time_f:>8} {sy_status:>10} {ip_match:>9}")

    print("-" * len(hdr))
    print(f"\nSummary ({total} comparable instances):")
    print(f"  Exact match:    {matches}/{total}")
    print(f"  Sypha better:   {sypha_better}/{total}")
    print(f"  OR-Tools better: {ortools_better}/{total}")
    if total_ortools_time > 0:
        print(f"  Total OR-Tools IP time: {total_ortools_time:.1f}s")
    if total_sypha_time > 0:
        print(f"  Total Sypha time:       {total_sypha_time:.1f}s")

    # Save combined CSV
    combined_csv = results_dir / "combined_comparison.csv"
    with open(combined_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance", "or_lp_obj", "or_ip_obj", "or_ip_status", "or_ip_time_s",
            "sypha_incumbent", "sypha_mip_gap_pct", "sypha_status", "sypha_time_s", "comparison"
        ])
        for inst in common:
            ot = ortools[inst]
            sy = sypha[inst]
            or_ip = ot.get("ip_objective", "")
            sy_inc = sy.get("incumbent", "")
            comp = ""
            if or_ip and sy_inc:
                try:
                    if abs(float(or_ip) - float(sy_inc)) < 0.5:
                        comp = "MATCH"
                    elif float(sy_inc) < float(or_ip):
                        comp = "SYPHA_BETTER"
                    else:
                        comp = "ORTOOLS_BETTER"
                except ValueError:
                    comp = "ERROR"
            writer.writerow([
                inst,
                ot.get("lp_objective", ""),
                or_ip,
                ot.get("ip_status", ""),
                ot.get("ip_solve_time", ""),
                sy_inc,
                sy.get("mip_gap_pct", ""),
                sy.get("status", ""),
                sy.get("time_total_s", ""),
                comp,
            ])
    print(f"\nCombined CSV saved to {combined_csv}")


if __name__ == "__main__":
    main()
