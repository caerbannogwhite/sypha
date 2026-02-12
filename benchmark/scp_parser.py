"""
SCP (Set Cover Problem) file format parser.
Format:
  Line 1: num_sets num_elements
  Line 2: cost_1 cost_2 ... cost_num_sets
  For each set i:
    Line: num_elements_in_set element_1 element_2 ... element_k
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

    # Parse first line: num_sets num_elements
    first_line = lines[0].split()
    num_sets = int(first_line[0])
    num_elements = int(first_line[1])

    # Parse costs (may span multiple lines)
    costs = []
    line_idx = 1
    while len(costs) < num_sets and line_idx < len(lines):
        costs.extend(map(float, lines[line_idx].split()))
        line_idx += 1

    # Trim to exact number of sets (in case we read too many)
    costs = costs[:num_sets]

    # Parse sets
    sets = []
    for _ in range(num_sets):
        if line_idx >= len(lines):
            sets.append([])
            continue

        parts = list(map(int, lines[line_idx].split()))
        if not parts:
            sets.append([])
        else:
            count = parts[0]
            elements = parts[1 : count + 1] if len(parts) > 1 else []

            # Handle case where elements span multiple lines
            elements_read = len(elements)
            line_idx += 1
            while elements_read < count and line_idx < len(lines):
                next_parts = list(map(int, lines[line_idx].split()))
                elements.extend(next_parts)
                elements_read += len(next_parts)
                line_idx += 1

            sets.append(elements[:count])
            continue

        line_idx += 1

    return {
        "num_sets": num_sets,
        "num_elements": num_elements,
        "costs": costs,
        "sets": sets,
    }
