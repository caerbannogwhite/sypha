#include "sypha/sypha.h"

#include "sypha_environment.h"
#include "sypha_node_sparse.h"
#include "sypha_solver_sparse.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Variable
// ---------------------------------------------------------------------------

namespace sypha {

Variable::Variable(int index, double lb, double ub, bool integer, const std::string& name)
    : name_(name), index_(index), lb_(lb), ub_(ub), integer_(integer) {}

const std::string& Variable::name() const { return name_; }
double Variable::solution_value() const { return solution_value_; }
double Variable::lb() const { return lb_; }
double Variable::ub() const { return ub_; }
int Variable::index() const { return index_; }
bool Variable::integer() const { return integer_; }

// ---------------------------------------------------------------------------
// Constraint
// ---------------------------------------------------------------------------

Constraint::Constraint(int index, double lb, double ub, const std::string& name)
    : name_(name), index_(index), lb_(lb), ub_(ub) {}

const std::string& Constraint::name() const { return name_; }
double Constraint::lb() const { return lb_; }
double Constraint::ub() const { return ub_; }
double Constraint::dual_value() const { return dual_value_; }

void Constraint::SetCoefficient(const Variable* var, double coeff) {
    for (auto& p : coeffs_) {
        if (p.first == var->index()) {
            p.second = coeff;
            return;
        }
    }
    coeffs_.emplace_back(var->index(), coeff);
}

double Constraint::GetCoefficient(const Variable* var) const {
    for (const auto& p : coeffs_) {
        if (p.first == var->index()) return p.second;
    }
    return 0.0;
}

void Constraint::SetBounds(double lb, double ub) {
    lb_ = lb;
    ub_ = ub;
}

// ---------------------------------------------------------------------------
// Objective
// ---------------------------------------------------------------------------

void Objective::SetCoefficient(const Variable* var, double coeff) {
    for (auto& p : coeffs_) {
        if (p.first == var->index()) {
            p.second = coeff;
            return;
        }
    }
    coeffs_.emplace_back(var->index(), coeff);
}

double Objective::GetCoefficient(const Variable* var) const {
    for (const auto& p : coeffs_) {
        if (p.first == var->index()) return p.second;
    }
    return 0.0;
}

void Objective::SetMinimization() { maximize_ = false; }
void Objective::SetMaximization() { maximize_ = true; }
void Objective::SetOffset(double offset) { offset_ = offset; }
double Objective::Value() const { return value_; }
double Objective::BestBound() const { return best_bound_; }
void Objective::Clear() {
    coeffs_.clear();
    maximize_ = false;
    offset_ = 0.0;
    value_ = 0.0;
    best_bound_ = 0.0;
}

// ---------------------------------------------------------------------------
// SolverImpl
// ---------------------------------------------------------------------------

class SolverImpl {
public:
    std::string name_;
    std::vector<std::unique_ptr<Variable>> variables_;
    std::vector<std::unique_ptr<Constraint>> constraints_;
    Objective objective_;
    SolverParameters params_;

    double objective_value_ = 0.0;
    double dual_objective_value_ = 0.0;
    double mip_gap_ = std::numeric_limits<double>::infinity();
    int iterations_ = 0;
    int nodes_ = 0;
    double wall_time_ = 0.0;
    ResultStatus last_status_ = ResultStatus::NOT_SOLVED;

    bool hasIntegerVars() const {
        for (const auto& v : variables_) {
            if (v->integer()) return true;
        }
        return false;
    }

    // Build standard form: min c^T x, Ax = b, x >= 0
    struct StandardForm {
        int nrows;
        int ncols;
        int ncolsOriginal;  // user variables count
        std::vector<int> csrInds;
        std::vector<int> csrOffs;
        std::vector<double> csrVals;
        std::vector<double> obj;
        std::vector<double> rhs;
    };

    StandardForm buildStandardForm() const {
        StandardForm sf;
        const int n = (int)variables_.size();

        // First pass: count rows (ranges produce two rows)
        int numRows = 0;
        for (const auto& c : constraints_) {
            bool hasLb = std::isfinite(c->lb_);
            bool hasUb = std::isfinite(c->ub_);
            if (hasLb && hasUb && std::fabs(c->lb_ - c->ub_) > 1e-15) {
                numRows += 2;  // range constraint -> two rows
            } else {
                numRows += 1;
            }
        }

        // Count slack variables needed
        int numSlacks = 0;
        // Track which rows need slacks
        struct RowInfo {
            int constraintIdx;
            bool isGe;       // true = ">= lb" row, false = "<= ub" row (negated)
            bool isEquality;
            double rhsVal;
        };
        std::vector<RowInfo> rowInfos;
        rowInfos.reserve((size_t)numRows);

        for (int ci = 0; ci < (int)constraints_.size(); ++ci) {
            const auto& c = constraints_[ci];
            bool hasLb = std::isfinite(c->lb_);
            bool hasUb = std::isfinite(c->ub_);
            if (hasLb && hasUb && std::fabs(c->lb_ - c->ub_) <= 1e-15) {
                // Equality
                rowInfos.push_back({ci, true, true, c->lb_});
            } else if (hasLb && hasUb) {
                // Range: split into >= lb and <= ub
                rowInfos.push_back({ci, true, false, c->lb_});
                numSlacks++;
                rowInfos.push_back({ci, false, false, c->ub_});
                numSlacks++;
            } else if (hasLb) {
                // >= lb
                rowInfos.push_back({ci, true, false, c->lb_});
                numSlacks++;
            } else if (hasUb) {
                // <= ub (negate to get >= form)
                rowInfos.push_back({ci, false, false, c->ub_});
                numSlacks++;
            } else {
                // Free row (shouldn't normally happen)
                rowInfos.push_back({ci, true, true, 0.0});
            }
        }

        sf.nrows = numRows;
        sf.ncols = n + numSlacks;
        sf.ncolsOriginal = n;

        // Build objective
        sf.obj.resize((size_t)sf.ncols, 0.0);
        for (const auto& p : objective_.coeffs_) {
            if (p.first >= 0 && p.first < n) {
                double c = p.second;
                if (objective_.maximize_) c = -c;
                sf.obj[(size_t)p.first] = c;
            }
        }
        // Slack variables have zero cost (already initialized)

        // Build CSR
        sf.csrOffs.reserve((size_t)(numRows + 1));
        sf.csrOffs.push_back(0);
        sf.rhs.resize((size_t)numRows, 0.0);

        int slackCol = n;  // next slack column index

        for (int ri = 0; ri < numRows; ++ri) {
            const RowInfo& info = rowInfos[(size_t)ri];
            const auto& c = constraints_[(size_t)info.constraintIdx];

            if (info.isEquality) {
                // Ax = b: coefficients as-is, no slack
                for (const auto& p : c->coeffs_) {
                    sf.csrInds.push_back(p.first);
                    sf.csrVals.push_back(p.second);
                }
                sf.rhs[(size_t)ri] = info.rhsVal;
            } else if (info.isGe) {
                // >= lb: row coefficients as-is, add surplus (coeff -1), rhs = lb
                for (const auto& p : c->coeffs_) {
                    sf.csrInds.push_back(p.first);
                    sf.csrVals.push_back(p.second);
                }
                sf.csrInds.push_back(slackCol);
                sf.csrVals.push_back(-1.0);
                slackCol++;
                sf.rhs[(size_t)ri] = info.rhsVal;
            } else {
                // <= ub: negate all coefficients, add surplus (coeff -1), rhs = -ub
                for (const auto& p : c->coeffs_) {
                    sf.csrInds.push_back(p.first);
                    sf.csrVals.push_back(-p.second);
                }
                sf.csrInds.push_back(slackCol);
                sf.csrVals.push_back(-1.0);
                slackCol++;
                sf.rhs[(size_t)ri] = -info.rhsVal;
            }

            sf.csrOffs.push_back((int)sf.csrVals.size());
        }

        return sf;
    }

    ResultStatus solve() {
        StandardForm sf = buildStandardForm();

        // Set up SyphaEnvironment
        SyphaEnvironment env;
        env.setDefaultParameters();

        // Override with our parameters
        env.verbosityLevel = params_.verbosity;
        env.mehrotraMaxIter = params_.mehrotra_max_iter;
        env.bnbMaxNodes = params_.bnb_max_nodes;
        env.bnbHardTimeLimitSeconds = params_.bnb_hard_time_limit_sec;
        env.bnbLogIntervalSeconds = params_.bnb_log_interval_sec;
        env.bnbGapStagnationWindow = params_.bnb_gap_stagnation_window;
        env.bnbGapStallBranchIters = params_.bnb_gap_stall_iters;
        env.bnbGapStallMinImprovPct = params_.bnb_gap_stall_min_improv_pct;
        env.bnbIntegralityTol = params_.integrality_tol;
        env.bnbVarSelectionStrategy = params_.bnb_var_selection;
        env.bnbIntHeuristics = params_.bnb_heuristics;
        env.preprocessColumnStrategies = params_.preprocess_strategies;
        env.preprocessTimeLimitSeconds = params_.preprocess_time_limit_sec;
        env.bnbDisable = params_.disable_bnb;
        env.showSolution = params_.show_solution;
        env.modelType = MODEL_TYPE_SCP;
        env.inputFilePath = "";
        env.internalStatus = CODE_SUCCESFULL;

        // Init logger
        SyphaLogLevel logLevel;
        if (params_.verbosity <= 0)
            logLevel = LOG_ERROR;
        else if (params_.verbosity <= 5)
            logLevel = LOG_INFO;
        else if (params_.verbosity <= 15)
            logLevel = LOG_DEBUG;
        else
            logLevel = LOG_TRACE;

        env.logger_ = new SyphaLogger(env.timer(), logLevel);
        if (params_.bnb_hard_time_limit_sec > 0.0)
            env.logger_->setHardTimeLimit(params_.bnb_hard_time_limit_sec * 1000.0);

        // CUDA device setup
        SyphaStatus devStatus = env.setUpDevice();
        if (devStatus != CODE_SUCCESFULL) {
            last_status_ = ResultStatus::ABNORMAL;
            return last_status_;
        }

        // Create node and populate directly
        SyphaNodeSparse node(env);

        node.nrows = sf.nrows;
        node.ncols = sf.ncols;
        node.ncolsOriginal = sf.ncolsOriginal;
        node.ncolsInputOriginal = sf.ncolsOriginal;
        node.nnz = (int)sf.csrVals.size();

        *node.hCsrMatInds = std::move(sf.csrInds);
        *node.hCsrMatOffs = std::move(sf.csrOffs);
        *node.hCsrMatVals = std::move(sf.csrVals);

        // Objective & RHS: allocated with calloc to match destructor's free()
        node.hObjDns = (double*)calloc((size_t)sf.ncols, sizeof(double));
        std::memcpy(node.hObjDns, sf.obj.data(), sizeof(double) * (size_t)sf.ncols);

        node.hRhsDns = (double*)calloc((size_t)sf.nrows, sizeof(double));
        std::memcpy(node.hRhsDns, sf.rhs.data(), sizeof(double) * (size_t)sf.nrows);

        // Identity mapping for active-to-input columns
        node.hActiveToInputCols->resize((size_t)sf.ncolsOriginal);
        std::iota(node.hActiveToInputCols->begin(), node.hActiveToInputCols->end(), 0);

        // Copy model to GPU
        SyphaStatus copyStatus = node.copyModelOnDevice();
        if (copyStatus != CODE_SUCCESFULL) {
            last_status_ = ResultStatus::ABNORMAL;
            return last_status_;
        }

        double timeStart = env.timer();

        bool useLpPath = !hasIntegerVars() || params_.disable_bnb;

        if (useLpPath) {
            // LP path: call solver_sparse_mehrotra_run directly
            SolverExecutionConfig config;
            config.maxIterations = env.mehrotraMaxIter;
            config.gapStagnation.enabled = false;
            config.bnbNodeOrdinal = 0;
            config.denseSelectionLogEveryNodes = 1;

            SolverExecutionResult result;
            SyphaStatus status = solver_sparse_mehrotra_run(node, config, &result);

            iterations_ = result.iterations;
            nodes_ = 0;
            wall_time_ = (env.timer() - timeStart) / 1000.0;

            if (status != CODE_SUCCESFULL || result.status != CODE_SUCCESFULL) {
                if (result.terminationReason == SOLVER_TERM_INFEASIBLE_OR_NUMERICAL) {
                    last_status_ = ResultStatus::INFEASIBLE;
                } else {
                    last_status_ = ResultStatus::ABNORMAL;
                }
                objective_value_ = node.objvalPrim;
                dual_objective_value_ = node.objvalDual;
                mip_gap_ = result.relativeGap;
                return last_status_;
            }

            // Extract primal solution for user variables
            double objVal = 0.0;
            for (int j = 0; j < (int)variables_.size(); ++j) {
                double val = (j < (int)result.primalSolution.size()) ? result.primalSolution[(size_t)j] : 0.0;
                variables_[(size_t)j]->solution_value_ = val;
                objVal += sf.obj[(size_t)j] * val;
            }

            // Extract dual values for constraints
            for (int ci = 0; ci < (int)constraints_.size(); ++ci) {
                if (ci < (int)result.dualSolution.size()) {
                    constraints_[(size_t)ci]->dual_value_ = result.dualSolution[(size_t)ci];
                }
            }

            if (objective_.maximize_) {
                objective_value_ = -(objVal) + objective_.offset_;
                dual_objective_value_ = -(node.objvalDual) + objective_.offset_;
            } else {
                objective_value_ = objVal + objective_.offset_;
                dual_objective_value_ = node.objvalDual + objective_.offset_;
            }
            mip_gap_ = result.relativeGap;

            if (result.terminationReason == SOLVER_TERM_CONVERGED) {
                last_status_ = ResultStatus::OPTIMAL;
            } else {
                last_status_ = ResultStatus::FEASIBLE;
            }
        } else {
            // MIP path: call node.solve() which dispatches to BnB
            SyphaStatus status = node.solve();

            iterations_ = node.getIterations();
            wall_time_ = (env.timer() - timeStart) / 1000.0;

            if (status != CODE_SUCCESFULL) {
                last_status_ = ResultStatus::ABNORMAL;
                objective_value_ = node.objvalPrim;
                dual_objective_value_ = node.objvalDual;
                mip_gap_ = node.mipGap;
                return last_status_;
            }

            // BnB stores solution in node.hX with ncolsInputOriginal entries
            if (node.hX && std::isfinite(node.objvalPrim)) {
                for (int j = 0; j < (int)variables_.size(); ++j) {
                    double val = (j < node.ncolsInputOriginal) ? node.hX[j] : 0.0;
                    variables_[(size_t)j]->solution_value_ = val;
                }
            }

            if (objective_.maximize_) {
                objective_value_ = -(node.objvalPrim) + objective_.offset_;
                dual_objective_value_ = -(node.objvalDual) + objective_.offset_;
            } else {
                objective_value_ = node.objvalPrim + objective_.offset_;
                dual_objective_value_ = node.objvalDual + objective_.offset_;
            }
            mip_gap_ = node.mipGap;

            if (std::isfinite(node.objvalPrim)) {
                if (node.mipGap <= 0.0 || !std::isfinite(node.mipGap)) {
                    last_status_ = ResultStatus::OPTIMAL;
                } else {
                    last_status_ = ResultStatus::FEASIBLE;
                }
            } else {
                last_status_ = ResultStatus::INFEASIBLE;
            }
        }

        objective_.value_ = objective_value_;
        objective_.best_bound_ = dual_objective_value_;

        return last_status_;
    }
};

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

Solver::Solver(const std::string& name)
    : impl_(std::make_unique<SolverImpl>()) {
    impl_->name_ = name;
}

Solver::~Solver() = default;
Solver::Solver(Solver&&) noexcept = default;
Solver& Solver::operator=(Solver&&) noexcept = default;

Variable* Solver::MakeNumVar(double lb, double ub, const std::string& name) {
    int idx = (int)impl_->variables_.size();
    impl_->variables_.emplace_back(new Variable(idx, lb, ub, false, name));
    return impl_->variables_.back().get();
}

Variable* Solver::MakeIntVar(double lb, double ub, const std::string& name) {
    int idx = (int)impl_->variables_.size();
    impl_->variables_.emplace_back(new Variable(idx, lb, ub, true, name));
    return impl_->variables_.back().get();
}

Variable* Solver::MakeBoolVar(const std::string& name) {
    return MakeIntVar(0.0, 1.0, name);
}

Constraint* Solver::MakeRowConstraint(double lb, double ub, const std::string& name) {
    int idx = (int)impl_->constraints_.size();
    impl_->constraints_.emplace_back(new Constraint(idx, lb, ub, name));
    return impl_->constraints_.back().get();
}

Objective* Solver::MutableObjective() {
    return &impl_->objective_;
}

ResultStatus Solver::Solve() {
    return impl_->solve();
}

int Solver::num_variables() const { return (int)impl_->variables_.size(); }
int Solver::num_constraints() const { return (int)impl_->constraints_.size(); }
double Solver::objective_value() const { return impl_->objective_value_; }
double Solver::dual_objective_value() const { return impl_->dual_objective_value_; }
double Solver::mip_gap() const { return impl_->mip_gap_; }
int Solver::iterations() const { return impl_->iterations_; }
int Solver::nodes() const { return impl_->nodes_; }
double Solver::wall_time() const { return impl_->wall_time_; }

SolverParameters& Solver::parameters() { return impl_->params_; }
const SolverParameters& Solver::parameters() const { return impl_->params_; }

double Solver::infinity() { return std::numeric_limits<double>::infinity(); }

} // namespace sypha
