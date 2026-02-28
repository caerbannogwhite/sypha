#ifndef SYPHA_SYPHA_H
#define SYPHA_SYPHA_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sypha {

enum class ResultStatus {
    OPTIMAL,
    FEASIBLE,
    INFEASIBLE,
    NOT_SOLVED,
    ABNORMAL,
};

struct SolverParameters {
    int verbosity = 5;
    int mehrotra_max_iter = 25;
    int bnb_max_nodes = 100000;
    double bnb_hard_time_limit_sec = 0.0;
    double bnb_log_interval_sec = 5.0;
    int bnb_gap_stagnation_window = 50;
    int bnb_gap_stall_iters = 5;
    double bnb_gap_stall_min_improv_pct = 1.0;
    double integrality_tol = 1e-6;
    std::string bnb_var_selection = "most_fractional";
    std::string bnb_heuristics = "nearest_integer_fixing,dual_guided_cover_repair";
    std::string preprocess_strategies = "single_column_dominance,two_column_dominance";
    double preprocess_time_limit_sec = 5.0;
    bool disable_bnb = false;
    bool show_solution = false;

    // Linear solver strategy: "auto", "dense", "sparse_qr", "krylov"
    std::string linear_solver_strategy = "auto";
    int krylov_max_cg_iter = 500;
    double krylov_cg_tol_initial = 1e-2;
    double krylov_cg_tol_final = 1e-8;
    double krylov_cg_tol_decay_rate = 0.5;
};

class Solver;
class SolverImpl;

class Variable {
public:
    const std::string& name() const;
    double solution_value() const;
    double lb() const;
    double ub() const;
    int index() const;
    bool integer() const;

private:
    friend class Solver;
    friend class SolverImpl;
    Variable(int index, double lb, double ub, bool integer, const std::string& name);

    std::string name_;
    int index_;
    double lb_;
    double ub_;
    bool integer_;
    double solution_value_ = 0.0;
};

class Constraint {
public:
    const std::string& name() const;
    void SetCoefficient(const Variable* var, double coeff);
    double GetCoefficient(const Variable* var) const;
    void SetBounds(double lb, double ub);
    double lb() const;
    double ub() const;
    double dual_value() const;

private:
    friend class Solver;
    friend class SolverImpl;
    Constraint(int index, double lb, double ub, const std::string& name);

    std::string name_;
    int index_;
    double lb_;
    double ub_;
    std::vector<std::pair<int, double>> coeffs_;
    double dual_value_ = 0.0;
};

class Objective {
public:
    void SetCoefficient(const Variable* var, double coeff);
    double GetCoefficient(const Variable* var) const;
    void SetMinimization();
    void SetMaximization();
    void SetOffset(double offset);
    double Value() const;
    double BestBound() const;
    void Clear();

private:
    friend class Solver;
    friend class SolverImpl;

    std::vector<std::pair<int, double>> coeffs_;
    bool maximize_ = false;
    double offset_ = 0.0;
    double value_ = 0.0;
    double best_bound_ = 0.0;
};

class Solver {
public:
    explicit Solver(const std::string& name = "");
    ~Solver();
    Solver(Solver&&) noexcept;
    Solver& operator=(Solver&&) noexcept;

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    Variable* MakeNumVar(double lb, double ub, const std::string& name);
    Variable* MakeIntVar(double lb, double ub, const std::string& name);
    Variable* MakeBoolVar(const std::string& name);

    Constraint* MakeRowConstraint(double lb, double ub, const std::string& name = "");

    Objective* MutableObjective();

    ResultStatus Solve();

    int num_variables() const;
    int num_constraints() const;
    double objective_value() const;
    double dual_objective_value() const;
    double mip_gap() const;
    int iterations() const;
    int nodes() const;
    double wall_time() const;

    SolverParameters& parameters();
    const SolverParameters& parameters() const;

    static double infinity();

private:
    std::unique_ptr<SolverImpl> impl_;
};

} // namespace sypha

#endif // SYPHA_SYPHA_H
