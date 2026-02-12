
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

	sprintf(message, "PRIMAL: %30.20lf", mainNode->getObjvalPrim());
	env->logger(message, "INFO", 0);
	sprintf(message, "DUAL: %30.20lf", mainNode->getObjvalDual());
	env->logger(message, "INFO", 0);
	sprintf(message, "ITERATIONS: %d", mainNode->getIterations());
	env->logger(message, "INFO", 0);
	sprintf(message, "TIME START SOL: %15.6lf", mainNode->getTimeStartSol());
	env->logger(message, "INFO", 0);
	sprintf(message, "TIME PRE SOL: %15.6lf", mainNode->getTimePreSol());
	env->logger(message, "INFO", 0);
	sprintf(message, "TIME SOLVER: %15.6lf", mainNode->getTimeSolver());
	env->logger(message, "INFO", 0);
	sprintf(message, "TIME TOTAL:  %15.6lf", (env->timer() - timeStart));
	env->logger(message, "INFO", 0);

	return 0;
}
