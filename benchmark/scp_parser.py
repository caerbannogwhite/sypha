"""
SCP (Set Cover Problem) file format parser (OR-Library style).
Format:
  Line 1: num_elements num_sets
  Next tokens: cost_1 ... cost_num_sets
  For each element i:
    token: num_sets_covering_i followed by that many set indices (1-based)
"""


def parse_scp_file(filepath):
    """
    Parse an SCP file and return problem data.

    Returns:
        dict: {
            'num_sets': int,
            'num_elements': int,
            'costs': list of float,
            'sets': list of lists (each set contains element indices, 1-indexed)
        }
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse first line: num_elements num_sets (OR-Library)
    first_line = lines[0].split()
    num_elements = int(first_line[0])
    num_sets = int(first_line[1])

    # Parse costs (may span multiple lines)
    costs = []
    line_idx = 1
    while len(costs) < num_sets and line_idx < len(lines):
        costs.extend(map(float, lines[line_idx].split()))
        line_idx += 1

    # Trim to exact number of sets (in case we read too many)
    costs = costs[:num_sets]

    # Parse element rows and build set-wise incidence lists
    sets = [[] for _ in range(num_sets)]
    for elem in range(1, num_elements + 1):
        if line_idx >= len(lines):
            break
        parts = list(map(int, lines[line_idx].split()))
        if not parts:
            line_idx += 1
            continue

        count = parts[0]
        covering_sets = parts[1 : count + 1] if len(parts) > 1 else []

        # Handle case where indices span multiple lines
        values_read = len(covering_sets)
        line_idx += 1
        while values_read < count and line_idx < len(lines):
            next_parts = list(map(int, lines[line_idx].split()))
            covering_sets.extend(next_parts)
            values_read += len(next_parts)
            line_idx += 1

        for set_idx_1b in covering_sets[:count]:
            if 1 <= set_idx_1b <= num_sets:
                sets[set_idx_1b - 1].append(elem)

    return {
        "num_sets": num_sets,
        "num_elements": num_elements,
        "costs": costs,
        "sets": sets,
    }
