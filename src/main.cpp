
#include "common.h"
#include "main.h"
#include "timer.h"

int main(int argc, char *argv[])
{
	Parameters params;
	Instance inst;
	
	double start = GetTimer();
	comm_parse_program_args(params, argc, argv);

	comm_read_input_file(inst, params);

	double end = GetTimer();

	cout << "time = " << (end - start) << endl;

	return 0;
}
