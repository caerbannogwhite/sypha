#include <algorithm>
#include <cctype>
#include <boost/program_options.hpp>

#include "sypha_environment.h"
#include "sypha_environment_defaults.h"
#include "sypha_cuda_helper.h"

namespace po = boost::program_options;

SyphaEnvironment::SyphaEnvironment()
{
}

SyphaEnvironment::SyphaEnvironment(int argc, char *argv[])
{
    this->internalStatus = CODE_SUCCESSFUL;
    this->internalStatus = this->setDefaultParameters();
    if (this->internalStatus == CODE_SUCCESSFUL)
    {
        logger_ = std::make_unique<SyphaLogger>(this->timer(), LOG_INFO);
        this->internalStatus = this->readInputArguments(argc, argv);
        if (this->internalStatus == CODE_SUCCESSFUL)
        {
            SyphaLogLevel logLevel;
            if (verbosityLevel <= 0)
                logLevel = LOG_ERROR;
            else if (verbosityLevel <= 5)
                logLevel = LOG_INFO;
            else if (verbosityLevel <= 15)
                logLevel = LOG_DEBUG;
            else
                logLevel = LOG_TRACE;
            logger_->setVerbosity(logLevel);

            if (bnbHardTimeLimitSeconds > 0.0)
                logger_->setHardTimeLimit(bnbHardTimeLimitSeconds * 1000.0);

            this->internalStatus = this->setUpDevice();
        }
    }
}

SyphaEnvironment::~SyphaEnvironment()
{
    if (logger_)
    {
        logger_->flush();
    }
}

SyphaStatus SyphaEnvironment::setDefaultParameters()
{
    this->pxInfinity = sypha_environment_defaults::kPxInfinity;
    this->pxTolerance = sypha_environment_defaults::kPxTolerance;

    this->cudaDeviceId = sypha_environment_defaults::kCudaDeviceId;
    this->verbosityLevel = sypha_environment_defaults::kVerbosityLevel;
    this->mehrotraMaxIter = sypha_environment_defaults::kMehrotraMaxIter;
    this->mehrotraEta = sypha_environment_defaults::kMehrotraEta;
    this->mehrotraMuTol = sypha_environment_defaults::kMehrotraMuTol;
    this->mehrotraCholTol = sypha_environment_defaults::kMehrotraCholTol;
    this->mehrotraReorder = sypha_environment_defaults::kMehrotraReorder;
    this->denseGpuMemoryFractionThreshold = sypha_environment_defaults::kDenseGpuMemoryFractionThreshold;

    this->linearSolverStrategy = sypha_environment_defaults::kLinearSolverStrategy();
    this->krylovMaxCgIter = sypha_environment_defaults::kKrylovMaxCgIter;
    this->krylovCgTolInitial = sypha_environment_defaults::kKrylovCgTolInitial;
    this->krylovCgTolFinal = sypha_environment_defaults::kKrylovCgTolFinal;
    this->krylovCgTolDecayRate = sypha_environment_defaults::kKrylovCgTolDecayRate;

    this->bnbMaxNodes = sypha_environment_defaults::kBnbMaxNodes;
    this->bnbDeviceQueueCapacity = sypha_environment_defaults::kBnbDeviceQueueCapacity;
    this->bnbGapStallBranchIters = sypha_environment_defaults::kBnbGapStallBranchIters;
    this->bnbGapStallMinImprovPct = sypha_environment_defaults::kBnbGapStallMinImprovPct;
    this->bnbIntegralityTol = sypha_environment_defaults::kBnbIntegralityTol;
    this->bnbVarSelectionStrategy = sypha_environment_defaults::kBnbVarSelectionStrategy();
    this->bnbHeuristicEveryNNodes = sypha_environment_defaults::kBnbHeuristicEveryNNodes;
    this->bnbIntHeuristics = sypha_environment_defaults::kBnbIntHeuristics();
    this->bnbLogIntervalSeconds = sypha_environment_defaults::kBnbLogIntervalSeconds;
    this->bnbHardTimeLimitSeconds = sypha_environment_defaults::kBnbHardTimeLimitSeconds;
    this->bnbGapStagnationWindow = sypha_environment_defaults::kBnbGapStagnationWindow;
    this->bnbDisable = sypha_environment_defaults::kBnbDisable;
    this->bnbAutoFallbackLp = sypha_environment_defaults::kBnbAutoFallbackLp;
    this->showSolution = sypha_environment_defaults::kShowSolution;
    this->preprocessColumnStrategies = sypha_environment_defaults::kPreprocessColumnStrategies();
    this->preprocessTimeLimitSeconds = sypha_environment_defaults::kPreprocessTimeLimitSeconds;

    return CODE_SUCCESSFUL;
}

SyphaStatus SyphaEnvironment::setUpDevice()
{

    logger_->log(LOG_DEBUG, "Setting up CUDA device");
    this->cudaDeviceId = findCudaDevice(this->cudaDeviceId);

    return CODE_SUCCESSFUL;
}

SyphaStatus SyphaEnvironment::readInputArguments(int argc, char *argv[])
{
    std::string modelType;

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("unit-tests", po::value<std::string>(&this->test)->default_value("none"), "launch unit tests")
            ("unit-tests-rep", po::value<int>(&this->testRepeat)->default_value(1), "set number of repeats for each test")
            ("input-file", po::value<std::string>(&this->inputFilePath), "set input file path")
            ("model", po::value<std::string>(&modelType), "set input model type (scp)")
            ("sparse", po::value<bool>(&this->sparse)->default_value(true), "import model as sparse model")
            ("time-limit", po::value<double>(&this->timeLimit), "set time limit")
            ("seed", po::value<int>(&this->seed), "set random seed")
            ("thread", po::value<int>(&this->threadNum), "set number of thread")
            ("tol", po::value<double>(&this->pxTolerance)->default_value(1e-8), "set tolerance")
            ("verbosity", po::value<int>(&this->verbosityLevel)->default_value(5), "set verbosity level")
            ("debug", po::value<int>(&this->debugLevel)->default_value(0), "set debug level")
            ("show-solution", po::bool_switch(&this->showSolution)->default_value(sypha_environment_defaults::kShowSolution), "show final solution summary")
            ("mehrotra-max-iter", po::value<int>(&this->mehrotraMaxIter)->default_value(sypha_environment_defaults::kMehrotraMaxIter), "set max iterations for Mehrotra IPM")
            ("dense-memory-threshold", po::value<double>(&this->denseGpuMemoryFractionThreshold)->default_value(sypha_environment_defaults::kDenseGpuMemoryFractionThreshold), "use dense KKT solver when dense matrix bytes < this fraction of total GPU memory")
            ("linear-solver", po::value<std::string>(&this->linearSolverStrategy)->default_value(sypha_environment_defaults::kLinearSolverStrategy()), "linear solver strategy: auto|dense|sparse_qr|krylov")
            ("krylov-max-cg-iter", po::value<int>(&this->krylovMaxCgIter)->default_value(sypha_environment_defaults::kKrylovMaxCgIter), "max CG iterations for Krylov solver")
            ("krylov-cg-tol-initial", po::value<double>(&this->krylovCgTolInitial)->default_value(sypha_environment_defaults::kKrylovCgTolInitial), "initial CG relative tolerance")
            ("krylov-cg-tol-final", po::value<double>(&this->krylovCgTolFinal)->default_value(sypha_environment_defaults::kKrylovCgTolFinal), "final CG relative tolerance")
            ("krylov-cg-tol-decay", po::value<double>(&this->krylovCgTolDecayRate)->default_value(sypha_environment_defaults::kKrylovCgTolDecayRate), "CG tolerance decay rate per IPM iteration")
            ("disable-bnb", po::bool_switch(&this->bnbDisable)->default_value(sypha_environment_defaults::kBnbDisable), "disable branch-and-bound and solve LP relaxation only")
            ("bnb-auto-fallback-lp", po::value<bool>(&this->bnbAutoFallbackLp)->default_value(sypha_environment_defaults::kBnbAutoFallbackLp), "fallback to LP relaxation if BnB finds no incumbent within limits")
            ("bnb-max-nodes", po::value<int>(&this->bnbMaxNodes)->default_value(sypha_environment_defaults::kBnbMaxNodes), "set max number of BnB nodes to process")
            ("bnb-device-queue", po::value<int>(&this->bnbDeviceQueueCapacity)->default_value(sypha_environment_defaults::kBnbDeviceQueueCapacity), "set active BnB node queue capacity on device")
            ("bnb-gap-stall-iters", po::value<int>(&this->bnbGapStallBranchIters)->default_value(sypha_environment_defaults::kBnbGapStallBranchIters), "branch if primal/dual gap does not improve for this many iterations")
            ("bnb-gap-stall-pct", po::value<double>(&this->bnbGapStallMinImprovPct)->default_value(sypha_environment_defaults::kBnbGapStallMinImprovPct), "minimum gap improvement percentage to reset stall counter")
            ("bnb-int-tol", po::value<double>(&this->bnbIntegralityTol)->default_value(sypha_environment_defaults::kBnbIntegralityTol), "integrality tolerance for BnB")
            ("bnb-var-select", po::value<std::string>(&this->bnbVarSelectionStrategy)->default_value(sypha_environment_defaults::kBnbVarSelectionStrategy()), "variable selection strategy: most_fractional|highest_cost_fractional")
            ("bnb-int-heur-every", po::value<int>(&this->bnbHeuristicEveryNNodes)->default_value(sypha_environment_defaults::kBnbHeuristicEveryNNodes), "run integer heuristics every n BnB nodes")
            ("bnb-int-heuristics", po::value<std::string>(&this->bnbIntHeuristics)->default_value(sypha_environment_defaults::kBnbIntHeuristics()), "comma-separated integer heuristics")
            ("bnb-log-interval-sec", po::value<double>(&this->bnbLogIntervalSeconds)->default_value(sypha_environment_defaults::kBnbLogIntervalSeconds), "seconds between branch-and-bound progress logs (<=0 disables)")
            ("bnb-hard-time-limit-sec", po::value<double>(&this->bnbHardTimeLimitSeconds)->default_value(sypha_environment_defaults::kBnbHardTimeLimitSeconds), "hard time limit for branch-and-bound in seconds (<=0 disables)")
            ("bnb-gap-stagnation-window", po::value<int>(&this->bnbGapStagnationWindow)->default_value(sypha_environment_defaults::kBnbGapStagnationWindow), "reduce LP iterations when MIP gap stagnates for this many BnB nodes (<=0 disables)")
            ("preprocess-columns", po::value<std::string>(&this->preprocessColumnStrategies)->default_value(sypha_environment_defaults::kPreprocessColumnStrategies()), "comma-separated preprocessing rules: single_column_dominance,two_column_dominance,none")
            ("preprocess-time-limit-sec", po::value<double>(&this->preprocessTimeLimitSeconds)->default_value(sypha_environment_defaults::kPreprocessTimeLimitSeconds), "time limit in seconds for column dominance preprocessing (<=0 disables)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            return CODE_GENERIC_ERROR;
        }

        if (vm.count("input-file"))
        {
            logger_->log(LOG_DEBUG, "Input file: %s", vm["input-file"].as<std::string>().c_str());
        }
        else
        {
            logger_->log(LOG_ERROR, "Input file path not set");
            return CODE_GENERIC_ERROR;
        }
        if (vm.count("model"))
        {
            std::string modelStr = vm["model"].as<std::string>();
            std::transform(modelStr.begin(), modelStr.end(), modelStr.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            if (modelStr == "scp")
            {
                this->modelType = MODEL_TYPE_SCP;
            }
            else
            {
                logger_->log(LOG_ERROR, "Unsupported model type: %s", vm["model"].as<std::string>().c_str());
                return CODE_GENERIC_ERROR;
            }
            logger_->log(LOG_DEBUG, "Model type: %s", vm["model"].as<std::string>().c_str());
        }
        else
        {
            logger_->log(LOG_ERROR, "Input model type not set");
            return CODE_GENERIC_ERROR;
        }
        if (vm.count("time-limit"))
            logger_->log(LOG_DEBUG, "Time limit: %.3f s", vm["time-limit"].as<double>());
        if (vm.count("seed"))
            logger_->log(LOG_DEBUG, "Random seed: %d", vm["seed"].as<int>());
        if (vm.count("thread"))
            logger_->log(LOG_DEBUG, "Threads: %d", vm["thread"].as<int>());
        logger_->log(LOG_DEBUG, "Tolerance: %g", this->pxTolerance);
        logger_->log(LOG_DEBUG, "Debug level: %d", this->debugLevel);
        logger_->log(LOG_DEBUG, "Show solution: %s", this->showSolution ? "true" : "false");
        logger_->log(LOG_DEBUG, "Mehrotra max iterations: %d", this->mehrotraMaxIter);
        logger_->log(LOG_DEBUG, "Dense memory threshold: %g", this->denseGpuMemoryFractionThreshold);
        logger_->log(LOG_DEBUG, "Linear solver strategy: %s", this->linearSolverStrategy.c_str());
        logger_->log(LOG_DEBUG, "Krylov max CG iterations: %d", this->krylovMaxCgIter);
        logger_->log(LOG_DEBUG, "Krylov CG tol initial: %g", this->krylovCgTolInitial);
        logger_->log(LOG_DEBUG, "Krylov CG tol final: %g", this->krylovCgTolFinal);
        logger_->log(LOG_DEBUG, "Krylov CG tol decay: %g", this->krylovCgTolDecayRate);
        logger_->log(LOG_DEBUG, "BnB max nodes: %d", this->bnbMaxNodes);
        logger_->log(LOG_DEBUG, "BnB device queue capacity: %d", this->bnbDeviceQueueCapacity);
        logger_->log(LOG_DEBUG, "BnB gap stall iterations: %d", this->bnbGapStallBranchIters);
        logger_->log(LOG_DEBUG, "BnB gap stall min improvement: %.4f%%", this->bnbGapStallMinImprovPct);
        logger_->log(LOG_DEBUG, "BnB integrality tolerance: %g", this->bnbIntegralityTol);
        logger_->log(LOG_DEBUG, "BnB variable selection: %s", this->bnbVarSelectionStrategy.c_str());
        logger_->log(LOG_DEBUG, "BnB heuristic frequency: every %d nodes", this->bnbHeuristicEveryNNodes);
        logger_->log(LOG_DEBUG, "BnB integer heuristics: %s", this->bnbIntHeuristics.c_str());
        logger_->log(LOG_DEBUG, "BnB log interval: %.1f s", this->bnbLogIntervalSeconds);
        logger_->log(LOG_DEBUG, "BnB hard time limit: %.1f s", this->bnbHardTimeLimitSeconds);
        logger_->log(LOG_DEBUG, "BnB gap stagnation window: %d nodes", this->bnbGapStagnationWindow);
        logger_->log(LOG_DEBUG, "Preprocess columns: %s", this->preprocessColumnStrategies.c_str());
        logger_->log(LOG_DEBUG, "Preprocess time limit: %.1f s", this->preprocessTimeLimitSeconds);
        logger_->log(LOG_DEBUG, "BnB disabled: %s", this->bnbDisable ? "true" : "false");
        logger_->log(LOG_DEBUG, "BnB auto fallback LP: %s", this->bnbAutoFallbackLp ? "true" : "false");
        this->denseGpuMemoryFractionThreshold = std::max(0.0, this->denseGpuMemoryFractionThreshold);
    }
    catch (std::exception &e)
    {
        if (logger_)
            logger_->log(LOG_ERROR, "Argument parsing failed: %s", e.what());
        else
            fprintf(stderr, "error: %s\n", e.what());
        return CODE_GENERIC_ERROR;
    }
    catch (...)
    {
        if (logger_)
            logger_->log(LOG_ERROR, "Argument parsing failed: unknown exception");
        else
            fprintf(stderr, "Exception of unknown type!\n");
        return CODE_GENERIC_ERROR;
    }

    return CODE_SUCCESSFUL;
}

SyphaLogger *SyphaEnvironment::getLogger() const
{
    return logger_.get();
}

int SyphaEnvironment::getVerbosityLevel() const
{
    return this->verbosityLevel;
}

bool SyphaEnvironment::getShowSolution() const
{
    return this->showSolution;
}

std::string SyphaEnvironment::getTest() const
{
    return this->test;
}

SyphaStatus SyphaEnvironment::getStatus() const
{
    return this->internalStatus;
}

double SyphaEnvironment::timer() const
{
    using namespace std::chrono;
    auto now = steady_clock::now();
    return duration<double, std::milli>(now.time_since_epoch()).count();
}

ModelInputType SyphaEnvironment::getModelType() const { return this->modelType; }
const std::string &SyphaEnvironment::getInputFilePath() const { return this->inputFilePath; }
double SyphaEnvironment::getPxInfinity() const { return this->pxInfinity; }
double SyphaEnvironment::getPxTolerance() const { return this->pxTolerance; }

int SyphaEnvironment::getMehrotraMaxIter() const { return this->mehrotraMaxIter; }
double SyphaEnvironment::getMehrotraEta() const { return this->mehrotraEta; }
double SyphaEnvironment::getMehrotraMuTol() const { return this->mehrotraMuTol; }
double SyphaEnvironment::getMehrotraCholTol() const { return this->mehrotraCholTol; }
int SyphaEnvironment::getMehrotraReorder() const { return this->mehrotraReorder; }
double SyphaEnvironment::getDenseGpuMemoryFractionThreshold() const { return this->denseGpuMemoryFractionThreshold; }

const std::string &SyphaEnvironment::getLinearSolverStrategy() const { return this->linearSolverStrategy; }
int SyphaEnvironment::getKrylovMaxCgIter() const { return this->krylovMaxCgIter; }
double SyphaEnvironment::getKrylovCgTolInitial() const { return this->krylovCgTolInitial; }
double SyphaEnvironment::getKrylovCgTolFinal() const { return this->krylovCgTolFinal; }
double SyphaEnvironment::getKrylovCgTolDecayRate() const { return this->krylovCgTolDecayRate; }

int SyphaEnvironment::getBnbMaxNodes() const { return this->bnbMaxNodes; }
int SyphaEnvironment::getBnbDeviceQueueCapacity() const { return this->bnbDeviceQueueCapacity; }
int SyphaEnvironment::getBnbGapStallBranchIters() const { return this->bnbGapStallBranchIters; }
double SyphaEnvironment::getBnbGapStallMinImprovPct() const { return this->bnbGapStallMinImprovPct; }
double SyphaEnvironment::getBnbIntegralityTol() const { return this->bnbIntegralityTol; }
const std::string &SyphaEnvironment::getBnbVarSelectionStrategy() const { return this->bnbVarSelectionStrategy; }
int SyphaEnvironment::getBnbHeuristicEveryNNodes() const { return this->bnbHeuristicEveryNNodes; }
const std::string &SyphaEnvironment::getBnbIntHeuristics() const { return this->bnbIntHeuristics; }
double SyphaEnvironment::getBnbLogIntervalSeconds() const { return this->bnbLogIntervalSeconds; }
double SyphaEnvironment::getBnbHardTimeLimitSeconds() const { return this->bnbHardTimeLimitSeconds; }
int SyphaEnvironment::getBnbGapStagnationWindow() const { return this->bnbGapStagnationWindow; }
bool SyphaEnvironment::getBnbDisable() const { return this->bnbDisable; }
bool SyphaEnvironment::getBnbAutoFallbackLp() const { return this->bnbAutoFallbackLp; }

const std::string &SyphaEnvironment::getPreprocessColumnStrategies() const { return this->preprocessColumnStrategies; }
double SyphaEnvironment::getPreprocessTimeLimitSeconds() const { return this->preprocessTimeLimitSeconds; }