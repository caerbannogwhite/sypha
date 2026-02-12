
#include "main.h"

int main(int argc, char *argv[])
{
	char message[1024];

	SyphaEnvironment *env = new SyphaEnvironment(argc, argv);
	if (env->getStatus() != CODE_SUCCESFULL)
	{
		return -1;
	}

	if (env->getTest().compare("none") != 0)
	{
		test_launcher(*env);
		return 0;
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

	if (mainNode->readModel() != CODE_SUCCESFULL)
	{
		return 1;
	}

	if (mainNode->copyModelOnDevice() != CODE_SUCCESFULL)
	{
		return 1;
	}

	if (mainNode->solve() != CODE_SUCCESFULL)
	{
		return 1;
	}

	env->logger("--- Solution ---", "INFO", 0);
	sprintf(message, "  Primal:     %.20g", mainNode->getObjvalPrim());
	env->logger(message, "INFO", 0);
	sprintf(message, "  Dual:       %.20g", mainNode->getObjvalDual());
	env->logger(message, "INFO", 0);
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
