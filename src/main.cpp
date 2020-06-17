
#include "main.h"

int main(int argc, char *argv[])
{
	char message[1024];

	std::cout << "		                                       " << std::endl
			  << "		######  ##  #  ######  ##   #  ####    " << std::endl
			  << "		##      ##  #  ##   #  ##   #  ##  #   " << std::endl
			  << "		#####   ####   #####   ######  #####   " << std::endl
			  << "		    ##  ##     ##      ##   #  ##  ##  " << std::endl
			  << "		######  ##     ##      ##   #  ##  ##  " << std::endl
			  << "			                                   " << std::endl;

	SyphaEnvironment *env = new SyphaEnvironment(argc, argv);
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
