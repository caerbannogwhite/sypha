
#include "common.h"
#include "sypha_node_sparse.h"
#include "sypha_environment.h"

int main(int argc, char *argv[])
{
    auto env = std::make_unique<SyphaEnvironment>(argc, argv);
    if (env->getStatus() != CODE_SUCCESSFUL)
    {
        return -1;
    }

    SyphaLogger *log = env->getLogger();

    if (env->getVerbosityLevel() > 1)
    {
        std::cout << std::endl;
        std::cout << "	==================================================" << std::endl
                  << "	==      ===  ====  ==       ===  ====  =====  ====" << std::endl
                  << "	=  ====  ==   ==   ==  ====  ==  ====  ====    ===" << std::endl
                  << "	=  ====  ===  ==  ===  ====  ==  ====  ===  ==  ==" << std::endl
                  << "	==  ========  ==  ===  ====  ==  ====  ==  ====  =" << std::endl
                  << "	====  =======    ====       ===        ==  ====  =" << std::endl
                  << "	======  ======  =====  ========  ====  ==        =" << std::endl
                  << "	=  ====  =====  =====  ========  ====  ==  ====  =" << std::endl
                  << "	=  ====  =====  =====  ========  ====  ==  ====  =" << std::endl
                  << "	==      ======  =====  ========  ====  ==  ====  =" << std::endl
                  << "	=================================================="
                  << std::endl
                  << std::endl;
    }

    auto mainNode = std::make_unique<SyphaNodeSparse>(*env);

    log->log(LOG_INFO, "Environment initialized");
    double timeStart = env->timer();
    SyphaStatus runStatus = CODE_SUCCESSFUL;

    log->log(LOG_INFO, "Reading model");
    runStatus = mainNode->readModel();
    if (runStatus != CODE_SUCCESSFUL)
    {
        log->log(LOG_ERROR, "Model read failed with status %d", static_cast<int>(runStatus));
        return 1;
    }

    log->log(LOG_INFO, "Copying model to device");
    runStatus = mainNode->copyModelOnDevice();
    if (runStatus != CODE_SUCCESSFUL)
    {
        log->log(LOG_ERROR, "Device copy failed with status %d", static_cast<int>(runStatus));
        return 1;
    }

    log->log(LOG_INFO, "Launching solver");
    runStatus = mainNode->solve();
    if (runStatus != CODE_SUCCESSFUL)
    {
        log->log(LOG_ERROR, "Solver failed with status %d", static_cast<int>(runStatus));
        return 1;
    }

    log->log(LOG_INFO, "--- Solution ---");
    log->log(LOG_INFO, "  Primal:     %.20g", mainNode->getObjvalPrim());
    log->log(LOG_INFO, "  Dual:       %.20g", mainNode->getObjvalDual());
    if (std::isfinite(mainNode->getMipGap()))
        log->log(LOG_INFO, "  MIP gap:    %.6f%%", mainNode->getMipGap() * 100.0);
    else
        log->log(LOG_INFO, "  MIP gap:    n/a");

    log->log(LOG_INFO, "--- Run statistics ---");
    log->log(LOG_INFO, "  Iterations: %d", mainNode->getIterations());
    log->log(LOG_INFO, "  Time (s):   start %.3f  pre %.2f  solver %.2f  total %.2f",
        mainNode->getTimeStartSol() / 1000.0,
        mainNode->getTimePreSol() / 1000.0,
        mainNode->getTimeSolver() / 1000.0,
        (env->timer() - timeStart) / 1000.0);

    return 0;
}
