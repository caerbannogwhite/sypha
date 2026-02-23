#!/bin/bash
# Run Sypha solver on specified instances and collect results as CSV
# Usage: bash run_sypha_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYPHA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
OUTPUT_CSV="$RESULTS_DIR/sypha_results.csv"
TIME_LIMIT=120  # seconds per instance for BnB hard time limit

mkdir -p "$RESULTS_DIR"

# CSV header
echo "instance,num_sets,num_elements,primal,dual,mip_gap_pct,iterations,time_pre_s,time_solver_s,time_total_s,incumbent,status" > "$OUTPUT_CSV"

# Instances to benchmark
INSTANCES=()
for f in "$SYPHA_ROOT"/data/scp4*.txt "$SYPHA_ROOT"/data/scp5*.txt "$SYPHA_ROOT"/data/scpa*.txt "$SYPHA_ROOT"/data/scpb*.txt; do
    [ -f "$f" ] && INSTANCES+=("$(basename "$f")")
done

TOTAL=${#INSTANCES[@]}
echo "Running Sypha on $TOTAL instances (time limit: ${TIME_LIMIT}s each)"
echo "Results will be saved to $OUTPUT_CSV"
echo ""

# Change to repo root so docker compose finds the compose file
cd "$SYPHA_ROOT"

IDX=0
for INST in "${INSTANCES[@]}"; do
    IDX=$((IDX + 1))
    echo "[$IDX/$TOTAL] $INST ..."

    # Run Sypha via Docker with time limit; capture output
    RAW_OUTPUT=$(docker compose run --rm sypha \
        ./sypha --verbosity 5 --model SCP \
        --input-file "data/$INST" \
        --bnb-hard-time-limit-sec "$TIME_LIMIT" \
        2>&1)

    # Strip ANSI color codes
    OUTPUT=$(printf '%s' "$RAW_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g')

    # Parse output
    PRIMAL=$(printf '%s' "$OUTPUT" | grep -oP 'Primal:\s+\K[-+0-9.eE]+' | tail -1)
    DUAL=$(printf '%s' "$OUTPUT" | grep -oP 'Dual:\s+\K[-+0-9.eE]+' | tail -1)
    MIP_GAP=$(printf '%s' "$OUTPUT" | grep -oP 'MIP gap:\s+\K[0-9.]+' | tail -1)
    ITERS=$(printf '%s' "$OUTPUT" | grep -oP 'Iterations:\s+\K[0-9]+' | tail -1)
    TIME_PRE=$(printf '%s' "$OUTPUT" | grep -oP 'pre\s+\K[0-9.]+' | tail -1)
    TIME_SOLVER=$(printf '%s' "$OUTPUT" | grep -oP 'solver\s+\K[0-9.]+' | tail -1)
    TIME_TOTAL=$(printf '%s' "$OUTPUT" | grep -oP 'total\s+\K[0-9.]+' | tail -1)
    NSETS=$(printf '%s' "$OUTPUT" | grep -oP 'Original model:\s+\K[0-9]+(?=\s+rows)' | tail -1)
    NCOLS=$(printf '%s' "$OUTPUT" | grep -oP '[0-9]+(?=\s+columns)' | tail -1)

    # Get incumbent (best integer objective from BnB progress or final solution)
    INCUMBENT=$(printf '%s' "$OUTPUT" | grep -oP 'incumbent=\s*\K[0-9.]+' | tail -1)
    if [ -z "$INCUMBENT" ]; then
        INCUMBENT=$(printf '%s' "$OUTPUT" | grep -oP 'Preprocessing incumbent from[^:]+:\s+\K[0-9.]+' | tail -1)
    fi
    if [ -z "$INCUMBENT" ]; then
        INCUMBENT=$(printf '%s' "$OUTPUT" | grep -oP 'Greedy heuristic incumbent:\s+\K[0-9.]+' | tail -1)
    fi

    # Determine status
    if printf '%s' "$OUTPUT" | grep -q "Optimality proven"; then
        STATUS="OPTIMAL"
    elif printf '%s' "$OUTPUT" | grep -q "declaring optimal"; then
        STATUS="OPTIMAL"
    elif printf '%s' "$OUTPUT" | grep -q "Best integer incumbent found"; then
        STATUS="FEASIBLE"
    elif printf '%s' "$OUTPUT" | grep -q "No integer incumbent found"; then
        STATUS="NO_INCUMBENT"
    elif printf '%s' "$OUTPUT" | grep -q "Solver failed"; then
        STATUS="ERROR"
    else
        STATUS="UNKNOWN"
    fi

    # Default empty values
    [ -z "$PRIMAL" ] && PRIMAL=""
    [ -z "$DUAL" ] && DUAL=""
    [ -z "$MIP_GAP" ] && MIP_GAP=""
    [ -z "$ITERS" ] && ITERS=""
    [ -z "$TIME_PRE" ] && TIME_PRE=""
    [ -z "$TIME_SOLVER" ] && TIME_SOLVER=""
    [ -z "$TIME_TOTAL" ] && TIME_TOTAL=""
    [ -z "$NSETS" ] && NSETS=""
    [ -z "$NCOLS" ] && NCOLS=""
    [ -z "$INCUMBENT" ] && INCUMBENT=""

    echo "  Status=$STATUS  Incumbent=$INCUMBENT  MIPGap=${MIP_GAP}%  Time=${TIME_TOTAL}s"

    echo "$INST,$NCOLS,$NSETS,$PRIMAL,$DUAL,$MIP_GAP,$ITERS,$TIME_PRE,$TIME_SOLVER,$TIME_TOTAL,$INCUMBENT,$STATUS" >> "$OUTPUT_CSV"
done

echo ""
echo "Done. Results saved to $OUTPUT_CSV"
