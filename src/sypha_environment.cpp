#include <algorithm>
#include <cctype>
#include "sypha_environment.h"
#include "sypha_environment_defaults.h"
#include "sypha_cuda_helper.h"

SyphaEnvironment::SyphaEnvironment()
{
}

SyphaEnvironment::SyphaEnvironment(int argc, char *argv[])
{
    this->internalStatus = CODE_SUCCESFULL;
    this->internalStatus = this->setDefaultParameters();
    if (this->internalStatus == CODE_SUCCESFULL)
    {
        this->internalStatus = this->readInputArguments(argc, argv);
        if (this->internalStatus == CODE_SUCCESFULL)
        {
            this->internalStatus = this->setUpDevice();
        }
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
    this->bnbDisable = sypha_environment_defaults::kBnbDisable;
    this->bnbAutoFallbackLp = sypha_environment_defaults::kBnbAutoFallbackLp;
    this->showSolution = sypha_environment_defaults::kShowSolution;

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaEnvironment::setUpDevice()
{

    this->logger("Setting up device", "INFO", 2);
    this->cudaDeviceId = findCudaDevice(this->cudaDeviceId);

    return CODE_SUCCESFULL;
}

SyphaStatus SyphaEnvironment::readInputArguments(int argc, char *argv[])
{
    string modelType;

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()("help", "produce help message")("unit-tests", po::value<string>(&this->test)->default_value("none"), "launch unit tests")("unit-tests-rep", po::value<int>(&this->testRepeat)->default_value(1), "set number of repeats for each test")("input-file", po::value<string>(&this->inputFilePath), "set input file path")("model", po::value<string>(&modelType), "set input model type (scp)")("sparse", po::value<bool>(&this->sparse)->default_value(true), "import model as sparse model")("time-limit", po::value<double>(&this->timeLimit), "set time limit")("seed", po::value<int>(&this->seed), "set random seed")("thread", po::value<int>(&this->threadNum), "set number of thread")("tol", po::value<double>(&this->pxTolerance)->default_value(1e-8), "set tolerance")("verbosity", po::value<int>(&this->verbosityLevel)->default_value(5), "set verbosity level")("debug", po::value<int>(&this->debugLevel)->default_value(0), "set debug level")("show-solution", po::bool_switch(&this->showSolution)->default_value(sypha_environment_defaults::kShowSolution), "show final solution summary")("mehrotra-max-iter", po::value<int>(&this->mehrotraMaxIter)->default_value(sypha_environment_defaults::kMehrotraMaxIter), "set max iterations for Mehrotra IPM")("disable-bnb", po::bool_switch(&this->bnbDisable)->default_value(sypha_environment_defaults::kBnbDisable), "disable branch-and-bound and solve LP relaxation only")("bnb-auto-fallback-lp", po::value<bool>(&this->bnbAutoFallbackLp)->default_value(sypha_environment_defaults::kBnbAutoFallbackLp), "fallback to LP relaxation if BnB finds no incumbent within limits")("bnb-max-nodes", po::value<int>(&this->bnbMaxNodes)->default_value(sypha_environment_defaults::kBnbMaxNodes), "set max number of BnB nodes to process")("bnb-device-queue", po::value<int>(&this->bnbDeviceQueueCapacity)->default_value(sypha_environment_defaults::kBnbDeviceQueueCapacity), "set active BnB node queue capacity on device")("bnb-gap-stall-iters", po::value<int>(&this->bnbGapStallBranchIters)->default_value(sypha_environment_defaults::kBnbGapStallBranchIters), "branch if primal/dual gap does not improve for this many iterations")("bnb-gap-stall-pct", po::value<double>(&this->bnbGapStallMinImprovPct)->default_value(sypha_environment_defaults::kBnbGapStallMinImprovPct), "minimum gap improvement percentage to reset stall counter")("bnb-int-tol", po::value<double>(&this->bnbIntegralityTol)->default_value(sypha_environment_defaults::kBnbIntegralityTol), "integrality tolerance for BnB")("bnb-var-select", po::value<string>(&this->bnbVarSelectionStrategy)->default_value(sypha_environment_defaults::kBnbVarSelectionStrategy()), "variable selection strategy: most_fractional|highest_cost_fractional")("bnb-int-heur-every", po::value<int>(&this->bnbHeuristicEveryNNodes)->default_value(sypha_environment_defaults::kBnbHeuristicEveryNNodes), "run integer heuristics every n BnB nodes")("bnb-int-heuristics", po::value<string>(&this->bnbIntHeuristics)->default_value(sypha_environment_defaults::kBnbIntHeuristics()), "comma-separated integer heuristics")("bnb-log-interval-sec", po::value<double>(&this->bnbLogIntervalSeconds)->default_value(sypha_environment_defaults::kBnbLogIntervalSeconds), "seconds between branch-and-bound progress logs (<=0 disables)")("bnb-hard-time-limit-sec", po::value<double>(&this->bnbHardTimeLimitSeconds)->default_value(sypha_environment_defaults::kBnbHardTimeLimitSeconds), "hard time limit for branch-and-bound in seconds (<=0 disables)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            cout << desc << "\n";
            return CODE_GENERIC_ERROR;
        }

        if (vm.count("input-file"))
        {
            cout << "Input file path set to " << vm["input-file"].as<string>() << ".\n";
        }
        else
        {
            cout << "Input file path not set. Exiting.\n";
            return CODE_GENERIC_ERROR;
        }
        if (vm.count("model"))
        {
            string modelStr = vm["model"].as<string>();
            std::transform(modelStr.begin(), modelStr.end(), modelStr.begin(),
                           [](unsigned char c)
                           { return std::tolower(c); });
            if (modelStr == "scp")
            {
                this->modelType = MODEL_TYPE_SCP;
            }
            else
            {
                cout << "Unsupported model type. Exiting.\n";
                return CODE_GENERIC_ERROR;
            }
            cout << "Input model type set to " << vm["model"].as<string>() << ".\n";
        }
        else
        {
            cout << "Input model type not set. Exiting.\n";
            return CODE_GENERIC_ERROR;
        }
        if (vm.count("time-limit"))
        {
            cout << "Time limit set to " << vm["time-limit"].as<double>() << ".\n";
        }
        if (vm.count("seed"))
        {
            cout << "Random seed set to " << vm["seed"].as<int>() << ".\n";
        }
        if (vm.count("thread"))
        {
            cout << "Number of threads set to " << vm["thread"].as<int>() << ".\n";
        }
        if (vm.count("tol"))
        {
            cout << "Tolerance set to " << vm["tol"].as<double>() << ".\n";
        }
        if (vm.count("debug"))
        {
            cout << "Debug level set to " << vm["debug"].as<int>() << ".\n";
        }
        cout << "Show solution set to " << (this->showSolution ? "true" : "false") << ".\n";
        if (vm.count("mehrotra-max-iter"))
        {
            cout << "Mehrotra max iterations set to " << vm["mehrotra-max-iter"].as<int>() << ".\n";
        }
        if (vm.count("bnb-max-nodes"))
        {
            cout << "BnB max nodes set to " << vm["bnb-max-nodes"].as<int>() << ".\n";
        }
        if (vm.count("bnb-device-queue"))
        {
            cout << "BnB device queue capacity set to " << vm["bnb-device-queue"].as<int>() << ".\n";
        }
        if (vm.count("bnb-gap-stall-iters"))
        {
            cout << "BnB gap stall iterations set to " << vm["bnb-gap-stall-iters"].as<int>() << ".\n";
        }
        if (vm.count("bnb-gap-stall-pct"))
        {
            cout << "BnB gap stall min improvement percentage set to " << vm["bnb-gap-stall-pct"].as<double>() << ".\n";
        }
        if (vm.count("bnb-int-tol"))
        {
            cout << "BnB integrality tolerance set to " << vm["bnb-int-tol"].as<double>() << ".\n";
        }
        if (vm.count("bnb-var-select"))
        {
            cout << "BnB variable selection strategy set to " << vm["bnb-var-select"].as<string>() << ".\n";
        }
        if (vm.count("bnb-int-heur-every"))
        {
            cout << "BnB integer heuristic frequency set to " << vm["bnb-int-heur-every"].as<int>() << ".\n";
        }
        if (vm.count("bnb-int-heuristics"))
        {
            cout << "BnB integer heuristics set to " << vm["bnb-int-heuristics"].as<string>() << ".\n";
        }
        if (vm.count("bnb-log-interval-sec"))
        {
            cout << "BnB log interval set to " << vm["bnb-log-interval-sec"].as<double>() << " seconds.\n";
        }
        if (vm.count("bnb-hard-time-limit-sec"))
        {
            cout << "BnB hard time limit set to " << vm["bnb-hard-time-limit-sec"].as<double>() << " seconds.\n";
        }
        cout << "BnB disabled set to " << (this->bnbDisable ? "true" : "false") << ".\n";
        cout << "BnB auto fallback to LP set to " << (this->bnbAutoFallbackLp ? "true" : "false") << ".\n";
    }
    catch (exception &e)
    {
        cerr << "error: " << e.what() << "\n";
        return CODE_GENERIC_ERROR;
    }
    catch (...)
    {
        cerr << "Exception of unknown type!\n";
        return CODE_GENERIC_ERROR;
    }

    return CODE_SUCCESFULL;
}

void SyphaEnvironment::logger(string message, string type, int level)
{
    const string colorEnd = "\e[39m";
    const string colorGreen = "\e[32m";
    const string colorRed = "\e[31m";

    if (type == "ERROR")
    {
        cout << "\e[1m" << colorRed << "[ERROR]" << colorEnd << " " << message << "\e[21m" << std::endl;
    }
    else if (type == "INFO" && level < this->verbosityLevel)
    {
        cout << colorGreen << "[INFO]" << colorEnd << " " << message << std::endl;
    }
}

int SyphaEnvironment::getVerbosityLevel()
{
    return this->verbosityLevel;
}

bool SyphaEnvironment::getShowSolution()
{
    return this->showSolution;
}

std::string SyphaEnvironment::getTest()
{
    return this->test;
}

SyphaStatus SyphaEnvironment::getStatus()
{
    return this->internalStatus;
}

double SyphaEnvironment::timer()
{
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerElapsed, NULL);
    // timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec * 1000.0 + timerElapsed.tv_usec / 1000.0;
}