
#include "sypha_environment.h"
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

    this->PX_INFINITY = 1e50;
    this->PX_TOLERANCE = 1e-12;

    this->cudaDeviceId = -1;
    this->verbosityLevel = 5;

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
        desc.add_options()("help", "produce help message")("unit-tests", po::value<string>(&this->test)->default_value("none"), "launch unit tests")("unit-tests-rep", po::value<int>(&this->testRepeat)->default_value(1), "set number of repeats for each test")("input-file", po::value<string>(&this->inputFilePath), "set input file path")("model", po::value<string>(&modelType), "set input model type (scp)")("sparse", po::value<bool>(&this->sparse)->default_value(true), "import model as sparse model")("time-limit", po::value<double>(&this->timeLimit), "set time limit")("seed", po::value<int>(&this->seed), "set random seed")("thread", po::value<int>(&this->threadNum), "set number of thread")("tol", po::value<double>(&this->PX_TOLERANCE)->default_value(1e-8), "set tolerance")("verbosity", po::value<int>(&this->verbosityLevel)->default_value(5), "set verbosity level")("debug", po::value<int>(&this->DEBUG_LEVEL)->default_value(0), "set debug level");

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
            if (vm["model"].as<string>() == "scp")
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