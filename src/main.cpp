
#include "main.h"

int main(int argc, char *argv[])
{
	SyphaEnvironment *env = new SyphaEnvironment(argc, argv);
	SyphaNode *mainNode = new SyphaNode(*env);

	double start = GetTimer();

	mainNode->importModel();



	double end = GetTimer();

	cout << "time = " << (end - start) << endl;

	return 0;
}
