
#include "common.h"

/*double comm_get_time_sec()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}*/

SyphaStatus comm_my_simplex_form(Instance &inst)
{
    inst.localObj = (double*)realloc(inst.localObj, (inst.ncols + inst.nrows) * sizeof(double));
    inst.localMat = (double*)realloc(inst.localMat, (inst.ncols + inst.nrows) * inst.nrows * sizeof(double));

    int i, j;

    for (j = inst.ncols; j < inst.ncols + inst.nrows; ++j)
    {
        inst.localObj[j] = 0.0;
    }

    for (i = inst.nrows - 1; i >= 0; i--)
    {
        for (j = 0; j < inst.ncols; ++j)
        {
            inst.localMat[i * (inst.ncols + inst.nrows) + j] = -inst.localMat[i * (inst.ncols) + j];
        }
    }

    for (i = 0; i < inst.nrows; ++i)
    {
        for (j = inst.ncols; j < inst.ncols + inst.nrows; ++j)
        {
            inst.localMat[i * (inst.ncols + inst.nrows) + j] = i == (j - inst.ncols) ? 1.0 : 0.0;
        }
    }

    inst.ncols = inst.ncols + inst.nrows;
    return CODE_SUCCESFULL;
}

SyphaStatus comm_parse_program_args(Parameters &params, int argc, char* argv[])
{
    params.PX_INFINITY = 1e50;
    //params.PX_TOLERANCE = 1e-4;

    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()("help", "produce help message")
            ("inputFile", po::value<string>(&params.inputFilePath), "set input file path")
            ("timeLimit", po::value<double>(&params.timeLimit), "set time limit")
            ("seed", po::value<int>(&params.seed), "set random seed")
            ("thread", po::value<int>(&params.threadNum), "set number of thread")
            ("tol", po::value<double>(&params.PX_TOLERANCE)->default_value(1e-8), "set tolerance")
            ("debug", po::value<int>(&params.DEBUG_LEVEL)->default_value(0), "set debug level")
            ;

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
    catch (exception& e)
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

SyphaStatus comm_read_input_file(Instance &inst, Parameters &params)
{
    int lineCounter = 0, currColNumber, colIdx = 0, rowIdx = 0, num;
    bool newRowFlag = false;

    FILE* inputFileHandler = fopen(&params.inputFilePath[0], "r");

    while (!feof(inputFileHandler))
    {
        // first row: number of rows and cols
        if (lineCounter == 0)
        {
            if (!fscanf(inputFileHandler, "%d %d", &inst.nrows, &inst.ncols))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }

            inst.hostObj = (double*)calloc(inst.ncols, sizeof(double));
            inst.hostMat = (double*)calloc(inst.nrows * inst.ncols, sizeof(double));
        }

        // costs
        else if (lineCounter > 0 && lineCounter <= inst.ncols)
        {
            if (!fscanf(inputFileHandler, "%d", &num))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            inst.hostObj[colIdx++] = num;

            if (lineCounter == inst.ncols)
            {
                colIdx = 0;
                rowIdx = 0;
                newRowFlag = true;
            }
        }

        // new row
        else if (newRowFlag)
        {
            if (!fscanf(inputFileHandler, "%d", &currColNumber))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            newRowFlag = false;
        }

        // entries
        else
        {
            // num: index of the column whose coefficient must be 1, 1 <= index <= num_cols
            if (!fscanf(inputFileHandler, "%d", &num))
            {
                perror("ERROR: readInstance on fscanf.");
                return CODE_ERROR;
            }
            inst.hostMat[rowIdx * inst.ncols + num - 1] = 1;
            ++colIdx;

            if (currColNumber == colIdx)
            {
                colIdx = 0;
                ++rowIdx;
                newRowFlag = true;
            }
        }

        ++lineCounter;
    }

    fclose(inputFileHandler);
    return CODE_SUCCESFULL;
}

SyphaStatus comm_free_instance(Instance  &inst)
{
    free(inst.hostObj);
    free(inst.hostMat);

    return CODE_SUCCESFULL;
}