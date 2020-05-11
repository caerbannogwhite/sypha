
#include "sypha_environment.h"

SyphaEnvironment::SyphaEnvironment()
{
    this->PX_INFINITY = 1e50;
    this->PX_TOLERANCE = 1e-12;
}

SyphaEnvironment::SyphaEnvironment(int argc, char *argv[])
{
    this->PX_INFINITY = 1e50;
    this->PX_TOLERANCE = 1e-12;

    this->readInputArguments(argc, argv);
}

bool SyphaEnvironment::getSparse()
{
    return this->sparse;
}

int SyphaEnvironment::getSeed()
{
    return this->seed;
}

int SyphaEnvironment::getThreadNum()
{
    return this->threadNum;
}

double SyphaEnvironment::getTimeLimit()
{
    return this->timeLimit;
}

SyphaStatus SyphaEnvironment::readInputArguments(int argc, char *argv[])
{
    string modelType;

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()
        ("help", "produce help message")
        ("inputFile", po::value<string>(&this->inputFilePath), "set input file path")
        ("model", po::value<string>(&modelType), "set input model type (scp)")
        ("sparse", po::value<bool>(&this->sparse)->default_value(true), "import model as sparse model")
        ("timeLimit", po::value<double>(&this->timeLimit), "set time limit")
        ("seed", po::value<int>(&this->seed), "set random seed")
        ("thread", po::value<int>(&this->threadNum), "set number of thread")
        ("tol", po::value<double>(&this->PX_TOLERANCE)->default_value(1e-8), "set tolerance")
        ("debug", po::value<int>(&this->DEBUG_LEVEL)->default_value(0), "set debug level");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            cout << desc << "\n";
            return CODE_ERROR;
        }

        if (vm.count("inputFile"))
        {
            cout << "Input file path set to " << vm["inputFile"].as<string>() << ".\n";
        }
        else
        {
            cout << "Input file path not set. Exiting.\n";
            return CODE_ERROR;
        }
        if (vm.count("model"))
        {
            if (vm["model"].as<string>().compare("scp"))
            {
                this->modelType = MODEL_TYPE_SCP;
            } else {
                cout << "Input model type not set. Exiting.\n";
                return CODE_ERROR;
            }
            cout << "Input model type set to " << vm["model"].as<string>() << ".\n";
        }
        else
        {
            cout << "Input model type not set. Exiting.\n";
            return CODE_ERROR;
        }
        if (vm.count("timeLimit"))
        {
            cout << "Time limit set to " << vm["timeLimit"].as<double>() << ".\n";
        }
        if (vm.count("seed"))
        {
            cout << "Random seed set to " << vm["seed"].as<int>() << ".\n";
        }
        if (vm.count("threads"))
        {
            cout << "Number of threads set to " << vm["threads"].as<int>() << ".\n";
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
        return CODE_ERROR;
    }
    catch (...)
    {
        cerr << "Exception of unknown type!\n";
        return CODE_ERROR;
    }

    return CODE_SUCCESFULL;
}
