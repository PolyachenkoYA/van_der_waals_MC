#include <iostream>

#include "VdW.h"

int main(int argc, char** argv) {
//    if(argc != 11){
//        printf("usage:\n%s   L   Temp   lmd   dl   N_atoms   Nt_max   init_gen_mode   to_remember_timeevol   verbose   seed\n", argv[0]);
//        return 1;
//    }

//    double L = atof(argv[1]);
//    double Temp = atof(argv[2]);
//    double lmd =  atof(argv[3]);
//    double dl = atof(argv[4]);
//	unsigned int N_atoms =atoi(argv[5]);
//	long Nt_max = atol(argv[6]);
//	int init_gen_mode = atoi(argv[7]);
//    int to_remember_timeevol = atoi(argv[8]);
//    int verbose = atoi(argv[9]);
//    int my_seed = atoi(argv[10]);

	double L;
	double Temp;
	double lmd;
	double dl;
	unsigned int N_atoms;
	long Nt_max;
	int init_gen_mode;
	int to_remember_timeevol;
	int verbose;
	int my_seed;

	verbose = 2;
	my_seed = 2;

	L = 20;   // [sgm]
	Temp = 0.5;   // [eps]
	lmd = 2;   // [sgm]
	dl = 0.1;   // [sgm]
	N_atoms = VdW::powi(5, 3);
	init_gen_mode = gen_mode_ID_cubic_lattice;
	to_remember_timeevol = 1;
	Nt_max = 1000;

	// for valgrind
	Nt_max = 1000;
	
    int i, j;
	long OP_arr_len = 128;

	double *E;
	int *biggest_cluster_sizes;
    if(to_remember_timeevol){
		E = (double*) malloc(sizeof(double) * OP_arr_len);
		biggest_cluster_sizes = (int*) malloc(sizeof(int) * OP_arr_len);
    }

	//    printf("0: %d\n", VdW::get_seed_C());
    VdW::init_rand_C(my_seed);
//    printf("1: %d\n", VdW::get_seed_C());

	long Nt = 0;

//	VdW::run_FFS_C(&flux0, &d_flux0, L, Temp, h, states, N_init_states,
//			  Nt, to_remember_timeevol ? &OP_arr_len : nullptr, OP_interfaces, N_OP_interfaces,
//			  probs, d_probs, &E, &M, &biggest_cluster_sizes, verbose, init_gen_mode,
//			  interface_mode, def_spin_state);
	VdW::run_bruteforce_C(L, Temp, lmd, dl, N_atoms, Nt_max, init_gen_mode, &OP_arr_len, &Nt, &E, &biggest_cluster_sizes, verbose);

    if(to_remember_timeevol){
		free(E);   // the pointer to the array
		free(biggest_cluster_sizes);
    }

	printf("DONE\n");

    return 0;
}
