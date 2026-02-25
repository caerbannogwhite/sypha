#include "sypha/sypha.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scp_file> [verbosity]" << std::endl;
        return 1;
    }

    const char* filename = argv[1];
    int verbosity = (argc >= 3) ? std::atoi(argv[2]) : 5;

    // Read SCP file
    FILE* fp = std::fopen(filename, "r");
    if (!fp) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return 1;
    }

    int nrows = 0, ncols = 0;
    if (std::fscanf(fp, "%d %d", &nrows, &ncols) != 2) {
        std::cerr << "Error: failed to read dimensions" << std::endl;
        std::fclose(fp);
        return 1;
    }

    std::vector<double> costs((size_t)ncols);
    for (int j = 0; j < ncols; ++j) {
        if (std::fscanf(fp, "%lf", &costs[(size_t)j]) != 1) {
            std::cerr << "Error: failed to read cost at column " << j << std::endl;
            std::fclose(fp);
            return 1;
        }
    }

    // Read rows: each row has a count followed by 1-indexed column indices
    std::vector<std::vector<int>> row_cols((size_t)nrows);
    for (int i = 0; i < nrows; ++i) {
        int count = 0;
        if (std::fscanf(fp, "%d", &count) != 1) {
            std::cerr << "Error: failed to read row " << i << " count" << std::endl;
            std::fclose(fp);
            return 1;
        }
        row_cols[(size_t)i].resize((size_t)count);
        for (int j = 0; j < count; ++j) {
            int idx = 0;
            if (std::fscanf(fp, "%d", &idx) != 1) {
                std::cerr << "Error: failed to read column index at row " << i << std::endl;
                std::fclose(fp);
                return 1;
            }
            row_cols[(size_t)i][(size_t)j] = idx - 1;  // convert to 0-indexed
        }
    }
    std::fclose(fp);

    // Build model
    sypha::Solver solver("SCP");
    solver.parameters().verbosity = verbosity;

    // Create binary variables
    std::vector<sypha::Variable*> vars((size_t)ncols);
    for (int j = 0; j < ncols; ++j) {
        vars[(size_t)j] = solver.MakeBoolVar("x" + std::to_string(j));
    }

    // Set covering constraints: sum of selected columns >= 1 for each row
    for (int i = 0; i < nrows; ++i) {
        sypha::Constraint* ct = solver.MakeRowConstraint(1.0, sypha::Solver::infinity(),
                                                         "row" + std::to_string(i));
        for (int col : row_cols[(size_t)i]) {
            ct->SetCoefficient(vars[(size_t)col], 1.0);
        }
    }

    // Objective: minimize total cost
    sypha::Objective* obj = solver.MutableObjective();
    obj->SetMinimization();
    for (int j = 0; j < ncols; ++j) {
        obj->SetCoefficient(vars[(size_t)j], costs[(size_t)j]);
    }

    std::cout << "Model: " << nrows << " rows, " << ncols << " columns" << std::endl;

    // Solve
    sypha::ResultStatus status = solver.Solve();

    // Print results
    const char* status_str = "UNKNOWN";
    switch (status) {
        case sypha::ResultStatus::OPTIMAL:    status_str = "OPTIMAL"; break;
        case sypha::ResultStatus::FEASIBLE:   status_str = "FEASIBLE"; break;
        case sypha::ResultStatus::INFEASIBLE: status_str = "INFEASIBLE"; break;
        case sypha::ResultStatus::NOT_SOLVED: status_str = "NOT_SOLVED"; break;
        case sypha::ResultStatus::ABNORMAL:   status_str = "ABNORMAL"; break;
    }

    std::cout << "--- Solution ---" << std::endl;
    std::cout << "  Status:     " << status_str << std::endl;
    std::cout << "  Objective:  " << solver.objective_value() << std::endl;
    std::cout << "  Dual bound: " << solver.dual_objective_value() << std::endl;

    if (std::isfinite(solver.mip_gap()))
        std::printf("  MIP gap:    %.6f%%\n", solver.mip_gap() * 100.0);
    else
        std::cout << "  MIP gap:    n/a" << std::endl;

    std::cout << "  Iterations: " << solver.iterations() << std::endl;
    std::printf("  Wall time:  %.3f s\n", solver.wall_time());

    if (status == sypha::ResultStatus::OPTIMAL || status == sypha::ResultStatus::FEASIBLE) {
        int selected = 0;
        for (int j = 0; j < ncols; ++j) {
            if (vars[(size_t)j]->solution_value() > 0.5) {
                ++selected;
            }
        }
        std::cout << "  Selected:   " << selected << " columns" << std::endl;

        if (verbosity > 10) {
            for (int j = 0; j < ncols; ++j) {
                if (vars[(size_t)j]->solution_value() > 0.5) {
                    std::printf("    x[%d] = 1 (cost %.1f)\n", j, costs[(size_t)j]);
                }
            }
        }
    }

    return 0;
}
