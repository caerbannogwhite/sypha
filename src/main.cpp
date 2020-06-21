
#include "main.h"

SyphaStatus launchTests(std::string test);

int main(int argc, char *argv[])
{
	char message[1024];

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
			  << "	==================================================\n" << std::endl;


	SyphaEnvironment *env = new SyphaEnvironment(argc, argv);
	if (env->getTest().compare("none") != 0)
	{
		launchTests(env->getTest());
		return 0;
	}

	SyphaNodeSparse *mainNode = new SyphaNodeSparse(*env);
	//SyphaNodeDense *mainNode = new SyphaNodeDense(*env);

	env->logger("Environment initialised", "INFO", 5);

	double timeStart = GetTimer();

	mainNode->readModel();
	mainNode->copyModelOnDevice();
	mainNode->solve();

	sprintf(message, "Total time: %lf ms", (GetTimer() - timeStart));
	env->logger(message, "INFO", 3);

	return 0;
}

SyphaStatus launchTests(std::string test)
{
	sypha_test_001();

	return CODE_SUCCESFULL;
}
