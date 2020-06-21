
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
	//SyphaNodeDense *mainNode = new SyphaNodeDense(*env);

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

	sprintf(message, "Solver time: %10.4lf ms", (mainNode->getTimeSolverEnd() - mainNode->getTimeSolverStart()));
	env->logger(message, "INFO", 5);
	sprintf(message, "Total time:  %10.4lf ms", (env->timer() - timeStart));
	env->logger(message, "INFO", 3);

	return 0;
}
