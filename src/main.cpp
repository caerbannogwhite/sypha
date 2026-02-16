
#include "main.h"

int main(int argc, char *argv[])
{
	char message[1024];

	SyphaEnvironment *env = new SyphaEnvironment(argc, argv);
	if (env->getStatus() != CODE_SUCCESFULL)
	{
		return -1;
	}

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

	SyphaNodeSparse *mainNode = new SyphaNodeSparse(*env);

	env->logger("Environment initialised", "INFO", 5);
	double timeStart = env->timer();
	SyphaStatus runStatus = CODE_SUCCESFULL;

	env->logger("Reading model", "INFO", 5);
	runStatus = mainNode->readModel();
	if (runStatus != CODE_SUCCESFULL)
	{
		sprintf(message, "readModel failed with status %d", (int)runStatus);
		env->logger(message, "ERROR", 0);
		return 1;
	}

	env->logger("Copying model to device", "INFO", 5);
	runStatus = mainNode->copyModelOnDevice();
	if (runStatus != CODE_SUCCESFULL)
	{
		sprintf(message, "copyModelOnDevice failed with status %d", (int)runStatus);
		env->logger(message, "ERROR", 0);
		return 1;
	}

	env->logger("Launching solver", "INFO", 5);
	runStatus = mainNode->solve();
	if (runStatus != CODE_SUCCESFULL)
	{
		sprintf(message, "solve failed with status %d", (int)runStatus);
		env->logger(message, "ERROR", 0);
		return 1;
	}

	env->logger("--- Solution ---", "INFO", 0);
	sprintf(message, "  Primal:     %.20g", mainNode->getObjvalPrim());
	env->logger(message, "INFO", 0);
	sprintf(message, "  Dual:       %.20g", mainNode->getObjvalDual());
	env->logger(message, "INFO", 0);
	if (std::isfinite(mainNode->getMipGap()))
	{
		sprintf(message, "  MIP gap:    %.6f%%", mainNode->getMipGap() * 100.0);
	}
	else
	{
		sprintf(message, "  MIP gap:    n/a");
	}
	env->logger(message, "INFO", 0);

	env->logger("--- Run statistics ---", "INFO", 0);
	sprintf(message, "  Iterations: %d", mainNode->getIterations());
	env->logger(message, "INFO", 0);
	sprintf(message, "  Time (s):   start %.3f  pre %.2f  solver %.2f  total %.2f",
		mainNode->getTimeStartSol() / 1000.0,
		mainNode->getTimePreSol() / 1000.0,
		mainNode->getTimeSolver() / 1000.0,
		(env->timer() - timeStart) / 1000.0);
	env->logger(message, "INFO", 0);

	return 0;
}
