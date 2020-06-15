
#include "main.h"

int main(int argc, char *argv[])
{
	SyphaEnvironment *env = new SyphaEnvironment(argc, argv);
	SyphaNodeSparse *mainNode = new SyphaNodeSparse(*env);

	double start = GetTimer();

	mainNode->importModel();
	mainNode->copyModelOnDevice();
	mainNode->solve();

	double end = GetTimer();

	cout << "time = " << (end - start) << endl;

	return 0;
}
