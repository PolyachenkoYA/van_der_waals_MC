//
// Created by ypolyach on 10/27/21.
//

#include <gsl/gsl_rng.h>
#include <cmath>
#include <optional>

#include <pybind11/pytypes.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <python3.8/Python.h>

#include <vector>

namespace py = pybind11;

#include "VdW.h"

py::int_ init_rand(int my_seed)
{
	VdW::init_rand_C(my_seed);
	return 0;
}

py::int_ get_seed()
{
	return VdW::seed;
}

py::int_ set_verbose(int new_verbose)
{
	VdW::verbose_dafault = new_verbose;
	return 0;
}

//py::tuple cluster_state(py::array_t<int> state, int default_state, std::optional<int> _verbose)
//{
//	py::buffer_info state_info = state.request();
//	int *state_ptr = static_cast<int *>(state_info.ptr);
//	assert(state_info.ndim == 1);
//
//	int L2 = state_info.shape[0];
//	int L = lround(sqrt(L2));
//	assert(L * L == L2);   // check for the full square
//
//	int verbose = (_verbose.has_value() ? _verbose.value() : VdW::verbose_dafault);
//	if(verbose > 5){
//		VdW::print_S(state_ptr, L, 'i');
//	}
//
//	py::array_t<int> cluster_element_inds = py::array_t<int>(L2);
//	py::buffer_info cluster_element_inds_info = cluster_element_inds.request();
//	int *cluster_element_inds_ptr = static_cast<int *>(cluster_element_inds_info.ptr);
//
//	py::array_t<int> cluster_sizes = py::array_t<int>(L2);
//	py::buffer_info cluster_sizes_info = cluster_sizes.request();
//	int *cluster_sizes_ptr = static_cast<int *>(cluster_sizes_info.ptr);
//
//	py::array_t<int> is_checked = py::array_t<int>(L2);
//	py::buffer_info is_checked_info = is_checked.request();
//	int *is_checked_ptr = static_cast<int *>(is_checked_info.ptr);
//
//	int N_clusters = L2;
//
//	VdW::clear_clusters(cluster_element_inds_ptr, cluster_sizes_ptr, &N_clusters);
//	VdW::uncheck_state(is_checked_ptr, L2);
//	VdW::cluster_state_C(state_ptr, L, cluster_element_inds_ptr, cluster_sizes_ptr, &N_clusters, is_checked_ptr, default_state);
//
//	return py::make_tuple(cluster_element_inds, cluster_sizes);
//}

py::tuple run_bruteforce(double L, int N_atoms, double Temp, double lmd, double dl, long Nt_max,
						 std::optional<int> _to_remember_timeevol, std::optional<int> _verbose, int gen_mode)
{
	int i, j;
//	double V = VdW::powi(L, dim);

// -------------- check input ----------------
	assert(L > 0);
	assert(Temp > 0);
	assert(lmd  > 0);
	assert((gen_mode == gen_mode_ID_cubic_lattice) || (gen_mode == gen_mode_ID_random));
	int verbose = (_verbose.has_value() ? _verbose.value() : VdW::verbose_dafault);
	int to_remember_timeevol = (_to_remember_timeevol.has_value() ? _to_remember_timeevol.value() : 1);

	double N_linear = pow(N_atoms, 1.0/dim);
	int N_linear_int = int(round(N_linear) + 0.1);
	if(VdW::powi(N_linear_int, dim) != N_atoms){
		printf("WARNING: N_atoms is not a cube of an integer, the filling will not be uniform\n");
	}

// ----------------- create return objects --------------
	long Nt = 0;
	long OP_arr_len = 128;   // the initial value that will be doubling when necessary

	double *_E;
	int *_biggest_cluster_sizes;
	if(to_remember_timeevol){
		_E = (double*) malloc(sizeof(double) * OP_arr_len);
		_biggest_cluster_sizes = (int*) malloc(sizeof(int) * OP_arr_len);
	}

	VdW::run_bruteforce_C(L, Temp, lmd, dl, N_atoms, Nt_max, gen_mode,
							to_remember_timeevol ? &OP_arr_len : nullptr,
							&Nt, &_E, &_biggest_cluster_sizes, verbose);

	int N_last_elements_to_print = std::min(Nt, (long)10);

	py::array_t<double> E;
	py::array_t<int> biggest_cluster_sizes;
	if(to_remember_timeevol){
		if(verbose >= 2){
			printf("Brute-force core done, Nt = %ld\n", Nt);
			VdW::print_E(&(_E[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'F');
//			VdW::print_biggest_cluster_sizes(&(_M[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'F');
		}

		E = py::array_t<double>(Nt);
		py::buffer_info E_info = E.request();
		double *E_ptr = static_cast<double *>(E_info.ptr);
		memcpy(E_ptr, _E, sizeof(double) * Nt);
		free(_E);

		biggest_cluster_sizes = py::array_t<int>(Nt);
		py::buffer_info biggest_cluster_sizes_info = biggest_cluster_sizes.request();
		int *biggest_cluster_sizes_ptr = static_cast<int *>(biggest_cluster_sizes_info.ptr);
		memcpy(biggest_cluster_sizes_ptr, _biggest_cluster_sizes, sizeof(int) * Nt);
		free(_biggest_cluster_sizes);

		if(verbose >= 2){
			printf("internal memory for EM freed\n");
			VdW::print_E(&(E_ptr[Nt - N_last_elements_to_print]), N_last_elements_to_print, 'P');
			printf("exiting py::run_bruteforce\n");
		}

	}

	return py::make_tuple(E, biggest_cluster_sizes);
}

namespace VdW
{
    gsl_rng *rng;
    int seed;
    int verbose_dafault;

	double md(double x, double L){ return x >= 0 ? (x < L ? x : x - L) : (L + x); }   // x mod L for x \in (-L;2L)
	double md_dx(double x, double R){ return x >= -R ? (x < R ? x : x - 2*R) : (2*R + x); }   // x mod L for x \in (-L;2L)
	void md_3D(double *X, double L){ crd3D_loop X[_ix] = md(X[_ix], L); }
	void md_dx_3D(double *X, double L){ crd3D_loop X[_ix] = md_dx(X[_ix], L); }
	void add_to_3D(double *X, const double *Y){ crd3D_loop X[_ix] += Y[_ix]; }
	void add_to_3D(double *X, double a){ crd3D_loop X[_ix] += a; }
	void substract_from_3D(double *X, const double *Y){ crd3D_loop X[_ix] -= Y[_ix]; }
	void mult_by_3D(double *X, double a){ crd3D_loop X[_ix] *= a; }
	void get_random_3D_point(double *X, double L){ crd3D_loop X[_ix] = gsl_rng_uniform(rng) * L; }
	void assign_to_3D(double *X, const double *Y){ memcpy(X, Y, sizeof(double ) * dim); }
	double dist2(const double *X1, const double *X2, double R, int m){
		double r2 = 0;
		if(m == 0) {
			crd3D_loop r2 += sqr(md_dx(X1[_ix] - X2[_ix], R));
		} else {
			double dr2;
			double dx;
			crd3D_loop {
				dx = md_dx(X1[_ix] - X2[_ix], R);
				printf("dx_%d = %lf\n", _ix, dx);
//				dr2 = sqr(dx);
				r2 += dx*dx;
			}
		}
		return r2;
	}
	double u_fnc(double r2, double lmd2){ return r2 < 1 ? 1e100 : (r2 < lmd2 ? -1 : 0); }

	int run_bruteforce_C(double L, double Temp, double lmd, double dl, int N_atoms, long Nt_max, int gen_mode,
						 long *OP_arr_len, long *Nt, double **E, int **biggest_cluster_sizes,
						 int verbose)
	{
//		double V = powi(L, dim);
		double lmd2 = sqr(lmd);
//		int state_size_in_bytes = sizeof(double) * N_atoms;

		int *cluster_element_inds = (int*) malloc(sizeof(int) * N_atoms);
		int *cluster_sizes = (int*) malloc(sizeof(int) * N_atoms);
		int *is_checked = (int*) malloc(sizeof(int) * N_atoms);

		double *X = (double*) malloc(sizeof(double) * dim * N_atoms);

		generate_state(X, L, N_atoms, gen_mode, verbose);
//		print_state(X, N_atoms, '0'); 		getchar();
		if(verbose){
			printf("running brute-force:\nL=%lf  T=%lf  lmd=%lf, dl=%lf  Nt_max=%ld  gen_mode=%d  verbose=%d  save_E=%d  save_CS=%d  to_resize_time_array=%d\n", L, Temp, lmd, dl, Nt_max, gen_mode, verbose, bool(E), bool(biggest_cluster_sizes), bool(OP_arr_len));
		}

		while(1){
			run_state(X, L, Temp, lmd2, dl * 2, N_atoms,
					  E, biggest_cluster_sizes, cluster_element_inds, cluster_sizes,
					  is_checked, Nt, OP_arr_len, verbose, Nt_max);
			// dl*2 because run_state uses dx \in [-dl/2; dl/2]

			if(Nt_max > 0){
				if(verbose)	{
					printf("brute-force done %lf              \r", (double)(*Nt) / Nt_max);
					fflush(stdout);
				}
				if(*Nt >= Nt_max) {
					if(verbose) printf("\n");
					break;
				}
			}
		}

		free(cluster_element_inds);
		free(cluster_sizes);
		free(is_checked);
		free(X);

		return 0;
	}

	int run_state(double *X, double L, double Temp, double lmd2, double dl, int N_atoms,
				  double **E, int **biggest_cluster_sizes,
				  int *cluster_element_inds, int *cluster_sizes, int *is_checked, long *Nt, long *OP_arr_len,
				  int verbose, long Nt_max)
	{
		int N_clusters_current = N_atoms;   // so that all uninitialized cluster_sizes are set to 0
		int biggest_cluster_sizes_current = 0;
		double R = L / 2;
		double E_current = comp_E(X, R, lmd2, N_atoms); // remember the 1st energy;
//		bool verbose_BF = (verbose < 0);
//		if(verbose_BF) verbose = -verbose;

		if(verbose >= 2){
			printf("E=%lf,  CS=%d\n", E_current, biggest_cluster_sizes_current);
		}

//		double Emin = -(2 + abs(h)) * L2;
//		double Emax = 2 * L2;
//		int default_spin_state = sgn(h);
//		double E_tolerance = 1e-3;   // J=1
//		long Nt_for_numerical_error = int(1e13 * E_tolerance / L2);
		// the error accumulates, so we need to recompute form scratch time to time
//      |E| ~ L2 => |dE_numerical| ~ L2 * 1e-13 => |dE_num_total| ~ sqrt(Nt * L2 * 1e-13) << E_tolerance => Nt << 1e13 * E_tolerance / L2
		double dE;
		int i_atom;
		while(1){
			// ----------- choose which to flip -----------
			dE = flip_random_particle(X, L, R, lmd2, dl, Temp, N_atoms, &i_atom);
//			if(dist2(&(X[52 * dim]), &(X[36 * dim]), L) < 1){
//				printf("Nt=%ld, i_atom=%d, dE=%lf, d2=%lf\n", *Nt, i_atom, dE, dist2(&(X[52 * dim]), &(X[36 * dim]), L));
//			}
//			if(*Nt > 11392){
//				printf("Nt=%ld, i_atom=%d, dE=%lf, d2=%lf\n", *Nt, i_atom, dE, dist2(&(X[52 * dim]), &(X[36 * dim]), L/2, 1));
//				print_X(&(X[52 * dim]));
//				print_X(&(X[36 * dim]));
//			}
//
//			if(dE < -N_atoms){
//				printf("Nt=%ld, dE=%lf, N_atoms=%d\n", *Nt, dE, N_atoms);
//				getchar();
//			}

			// --------------- compute time-dependent features ----------
			E_current += dE;

			clear_clusters(cluster_element_inds, cluster_sizes, &N_clusters_current);
			uncheck_state(is_checked, N_atoms);
			cluster_state_C(X, R, lmd2, N_atoms, cluster_element_inds, cluster_sizes, &N_clusters_current, is_checked);
			biggest_cluster_sizes_current = max(cluster_sizes, N_clusters_current);

			++(*Nt);

			if(verbose){
//				if(*Nt % (Nt_max / 1000 + 1) == 0){
//					fflush(stdout);
//				}
				if(!(*Nt % 10000)){
					if(Nt_max > 0){
						printf("BF run: %lf %%            \r", (double)(*Nt) / (Nt_max) * 100);
					} else {
						printf("BF run: Nt = %ld               \r", *Nt);
					}
					fflush(stdout);
				}
			}

			// ------------------ save timeevol ----------------
			if(OP_arr_len){
				if(*Nt >= *OP_arr_len){ // double the size of the time-index
					*OP_arr_len *= 2;
					if(E){
						*E = (double*) realloc (*E, sizeof(double) * *OP_arr_len);
						assert(*E);
					}
					if(biggest_cluster_sizes){
						*biggest_cluster_sizes = (int*) realloc (*biggest_cluster_sizes, sizeof(int) * *OP_arr_len);
						assert(*biggest_cluster_sizes);
					}

					if(verbose >= 2){
						printf("\nrealloced to %ld\n", *OP_arr_len);
					}
				}

				if(E) (*E)[*Nt - 1] = E_current;
				if(biggest_cluster_sizes) (*biggest_cluster_sizes)[*Nt - 1] = biggest_cluster_sizes_current;
			}
//			if(verbose >= 4) printf("done Nt=%d\n", *Nt-1);

			// ---------------- check exit ------------------
			if(Nt_max > 0){
				if(*Nt >= Nt_max){
					if(verbose){
						if(verbose >= 2) {
							printf("Reached desired Nt >= Nt_max (= %ld)\n", Nt_max);
						} else {
							printf("\n");
						}
					}

					return 1;
				}
			}
		}
	}

	void cluster_state_C(const double *X, double R, double lmd2, int N_atoms, int *cluster_element_inds, int *cluster_sizes, int *N_clusters, int *is_checked)
	{
		*N_clusters = 0;
		int N_clustered_elements = 0;
		for(int i = 0; i < N_atoms; ++i){
			if(!is_checked[i]){
				add_to_cluster(X, R, lmd2, N_atoms, is_checked, &(cluster_element_inds[N_clustered_elements]),
							   &(cluster_sizes[*N_clusters]), i, (*N_clusters) + 1);
				N_clustered_elements += cluster_sizes[*N_clusters];
				++(*N_clusters);
			}
		}

	}

	void add_to_cluster(const double* X, double R, double lmd2, int N_atoms, int* is_checked, int* cluster, int* cluster_size, int i_atom, int cluster_label)
	{
		if(!is_checked[i_atom]){
			is_checked[i_atom] = cluster_label;
			cluster[*cluster_size] = i_atom;
			++(*cluster_size);

			const double *X_atom = &(X[dim * i_atom]);
			for(int i = 0; i < N_atoms; ++i){
				if(dist2(X_atom, &(X[dim * i]), R) < lmd2) {
					add_to_cluster(X, R, lmd2, N_atoms, is_checked, cluster, cluster_size, i, cluster_label);
				}
			}
		}
	}

	void uncheck_state(int *is_checked, int N_atoms)
	{
		for(int i = 0; i < N_atoms; ++i) is_checked[i] = 0;
	}

	void clear_clusters(int *clusters, int *cluster_sizes, int *N_clusters)
	{
//		int N_done = 0;
		for(int i = 0; i < *N_clusters; ++i){
//			for(int j = 0; j < cluster_sizes[i]; ++j){
//				clusters[j + N_done] = -1;
//			}
//			N_done += cluster_sizes[i];
			cluster_sizes[i] = 0;
		}
		*N_clusters = 0;
	}

	double comp_E(const double * X, double R, double lmd2, int N_atoms)
	{
		int i, j, i3;
		double dr2;
		double E = 0;
		for(i = 0; i < N_atoms; ++i){
			i3 = i * dim;
			for(j = i + 1; j < N_atoms; ++j){
				dr2 = dist2(&(X[i3]), &(X[dim * j]), R);
				E += u_fnc(dr2, lmd2);
			}
		}

		return E;
	}

	bool X_fits_in(const double *X_new, const double *X_present, int N_atoms_present, double R, double r2_min)
	{
		for(int i = 0; i < N_atoms_present; ++i){
			if(dist2(X_new, &(X_present[i * dim]), R) <= r2_min){
				return false;
			}
		}
		return true;
	}

	void generate_state(double *X, double L, int N_atoms, int gen_mode, int verbose)
	{
		int i;
//		double V = powi(L, dim);
		int N_l = int(ceil(pow(N_atoms, 1.0/dim)) + 0.1);
		double l_step = L / N_l;
		double R = L / 2;

		switch (gen_mode) {
			case gen_mode_ID_random:
				// assert(\rho < \rho_dense_packing)
				double X_new[dim];
				for(i = 0; i < N_atoms; ++i){
					do{
						get_random_3D_point(X_new, L);
					}while(!X_fits_in(X_new, X, i, R));
					memcpy(&(X[i * dim]), X_new, sizeof(double) * dim);
				}
				break;
			case gen_mode_ID_cubic_lattice:
				int ix, iy, iz;
				for(ix = 0; ix < N_l; ++ix) for(iy = 0; iy < N_l; ++iy) for(iz = 0; iz < N_l; ++iz) {
					i = (iz + N_l * (iy + N_l * ix)) * dim;
					X[i] = ix * l_step;
					X[i + 1] = iy * l_step;
					X[i + 2] = iz * l_step;
				}
				break;
		}
	}

	double get_Ei(double *X, double R, double lmd2, int N_atoms, int i_atom, double *X_atom)
	{
		double E = 0;
		int i;
		double dr2;

		for(i = 0; i < i_atom; ++i){
			dr2 = dist2(&(X[i * dim]), X_atom, R);
//			if(dr2 < 1){
//				printf("BAD: i,j = %d,%d, dr2=%lf", i, i_atom, dr2);
//				getchar();
//			}
			E += u_fnc(dr2, lmd2);
		}
		for(i = i_atom + 1; i < N_atoms; ++i){
			dr2 = dist2(&(X[i * dim]), X_atom, R);
//			if(dr2 < 1){
//				printf("BAD: i,j = %d,%d, dr2=%lf", i, i_atom, dr2);
//				getchar();
//			}
			E += u_fnc(dr2, lmd2);
		}

		return E;
	}

	double get_dE_atom(double *X, double L, double R, double lmd2, int N_atoms, int i_atom, double *dX)
	{
		double *X_atom = &(X[dim * i_atom]);
		double E_old = get_Ei(X, R, lmd2, N_atoms, i_atom, X_atom);

		double X_new[dim];
		assign_to_3D(X_new, X_atom);
		add_to_3D(X_new, dX);
		md_3D(X_new, L);
		double E_new = get_Ei(X, R, lmd2, N_atoms, i_atom, X_new);

//		if(abs(E_old) > N_atoms){
//			printf("BAD: E_new=%lf, E_old=%lf\n", E_new, E_old);
//			getchar();
//		}

		return E_new - E_old;
	}

	double flip_random_particle(double *X, double L, double R, double lmd2, double dl, double Temp, int N_atoms, int* i_atom)
	{
//		int i_atom;
		double dX[dim];
		double dE;
		do{
			*i_atom = gsl_rng_uniform_int(rng, N_atoms);
			crd3D_loop dX[_ix] = (gsl_rng_uniform(rng) - 0.5) * dl;

			dE = get_dE_atom(X, L, R, lmd2, N_atoms, *i_atom, dX);

//			if(abs(dE) > N_atoms){
//				printf("i=%d, dE=%lf\n", *i_atom, dE);
//				getchar();
//			}
		}while(dE > N_atoms ? true : (dE > 0 ? (gsl_rng_uniform(rng) > exp(- dE / Temp)) : false));
//		!(dE <= 0 ? true : (gsl_rng_uniform(rng) < exp(- dE / Temp)))

//		if(abs(dE) > N_atoms){
//			printf("BAD: i=%d, dE=%lf\n", *i_atom, dE);
//			getchar();
//		}

		add_to_3D(&(X[*i_atom * dim]), dX);   // save the displacement
		md_3D(&(X[*i_atom * dim]), L);
		return dE;
	}

	int init_rand_C(int my_seed)
	/**
	 * Sets up a new GSL randomg denerator seed
	 * @param my_seed - the new seed for GSL
	 * @return  - the Error code
	 */
	{
		// initialize random generator
		gsl_rng_env_setup();
		const gsl_rng_type* T = gsl_rng_default;
		rng = gsl_rng_alloc(T);
		gsl_rng_set(rng, my_seed);
//		srand(my_seed);

		seed = my_seed;
		return 0;
	}

	void print_E(const double *E, long Nt, char prefix, char suffix)
    {
        if(prefix > 0) printf("Es: %c\n", prefix);
        for(int i = 0; i < Nt; ++i) printf("%lf ", E[i]);
        if(suffix > 0) printf("%c", suffix);
    }

	void print_X(const double *X, char prefix)
	{
		int i, j;

		if(prefix > 0){
			printf("%c", prefix);
		}
		printf("(%lf; %lf; %lf)\n", X[0], X[1], X[2]);
	}

    void print_state(const double *X, int N_atoms, char prefix)
    {
        int i, j;

        if(prefix > 0){
            printf("%c\n", prefix);
        }

        for(i = 0; i < N_atoms; ++i){
			j = dim * i;
//			printf("%4d: (%lf, %lf, %lf)\n", i, X[j], X[j + 1], X[j + 2]);
			printf("%4d: ", i);
			print_X(&(X[j]));
        }
    }

//	int get_init_states_C(int L, double Temp, double h, int N_init_states, int *init_states, int mode, int OP_thr_save_state,
//						  int interface_mode, int default_spin_state, int OP_A, int OP_B,
//						  double **E, int **M, int **biggest_cluster_size, int **h_A,
//						  long *Nt, long *OP_arr_len, int verbose)
//	/**
//	 *
//	 * @param L - see run_FFS
//	 * @param Temp - see run_FFS
//	 * @param h - see run_FFS
//	 * @param N_init_states - see run_FFS
//	 * @param OP_thr_save_state - see 'run_state' description
//	 * @param init_states - int*, assumed to be preallocated; The set of states to fill by generating them in this function
//	 * @param verbose - see run_FFS
//	 * @param mode - the mode of generation
//	 * 		mode >=0: generate all spins -1, then randomly (uniform ix and iy \in [0; L^2-1]) set |mode| states to +1
//	 * 		mode -1: all spins are 50/50 +1 or -1
//	 * 		mode -2: A brute-force simulation is run until 'N_init_states' states with M < M_0 are saved
//	 * 			The simluation is run in a following way:
//	 * 			1. A state with 'all spins = -1' is generated and saved, then a MC simulation is run until 'M_current' reaches ~ 0
//	 * 				(1 for odd L^2, 0 for even L^2. This is because M_step=2, so if L^2 is even, then all possible Ms are also even, and if L^2 is odd, all Ms are odd)
//	 * 			2. During the simulation, the state is saved every time when M < M_0
//	 * 			3. If we reach 'N_init_states' saved states before hitting 'M_current'~0, then the simulation just stops and the generation is complete
//	 * 			4. If we reach 'M_current'~0 before obtaining 'N_init_states' saved states, the simulation is restarted randomly from a state \in states that were already generated and saved
//	 * 			5. p. 4 in repeated until we obtain 'N_init_states' saved states
//	 * @return - the Error code
//	 */
//	{
//		int i;
//		int L2 = L*L;
////		int state_size_in_bytes = sizeof(int) * L2;
//
//		if(verbose){
//			printf("generating states:\nN_init_states=%d, gen_mode=%d, OP_A <= %d, OP_mode=%d\n", N_init_states, mode, OP_thr_save_state, interface_mode);
//		}
//
//		if(mode >= -1){
//			// generate N_init_states states in A
//			// Here they are identical, but I think it's better to generate them accordingly to equilibrium distribution in A
//			for(i = 0; i < N_init_states; ++i){
//				generate_state(&(init_states[i * L2]), L, mode, interface_mode, default_spin_state, verbose);
//			}
//		} else if(mode == -2){
//			int N_states_done;
//			int N_tries;
////			long Nt = 0;
//			run_bruteforce_C(L, Temp, h, N_init_states, init_states,
//							 nullptr, Nt, nullptr, nullptr, nullptr, nullptr,
//							 interface_mode, default_spin_state, 0, 0,
//							 OP_min_default[interface_mode], OP_B,
//							 &N_states_done, OP_min_default[interface_mode],
//							 OP_thr_save_state, save_state_mode_Inside, -1,
//							 verbose, -1, &N_tries);
//		} else if(mode == -3){
//			int N_states_done;
//			int N_tries;
////			long Nt = 0;
//			run_bruteforce_C(L, Temp, h, N_init_states, init_states,
//							 OP_arr_len, Nt, E, M, biggest_cluster_size, h_A,
//							 interface_mode, default_spin_state, OP_A, OP_B,
//							 OP_min_default[interface_mode], OP_B,
//							 &N_states_done, OP_thr_save_state - 1,
//							 OP_thr_save_state, save_state_mode_Influx, -1,
//							 verbose, -1, &N_tries);
//		}
//
//		return 0;
//	}
//
//	int is_infinite_cluster(const int* cluster, const int* cluster_size, int L, char *present_rows, char *present_columns)
//	{
//		int i = 0;
//
//		zero_array(present_rows, L);
//		zero_array(present_columns, L);
//		for(i = 0; i < (*cluster_size); ++i){
//			present_rows[cluster[i] / L] = 1;
//			present_columns[cluster[i] % L] = 1;
//		}
//		char cluster_is_infinite_x = 1;
//		char cluster_is_infinite_y = 1;
//		for(i = 0; i < L; ++i){
//			if(!present_columns[i]) cluster_is_infinite_x = 0;
//			if(!present_rows[i]) cluster_is_infinite_y = 0;
//		}
//
//		return cluster_is_infinite_x || cluster_is_infinite_y;
//	}
//
//	int state_is_valid(const int *s, int L, int k, char prefix)
//	{
//		for(int i = 0; i < L*L; ++i) if(abs(s[i]) != 1) {
//				printf("%d\n", k);
//				print_S(s, L, prefix);
//				return 0;
//			}
//		return 1;
//	}

//	int E_is_valid(const double *E, const double E1, const double E2, int N, int k, char prefix)
//	{
//		for(int i = 0; i < N; ++i) if((E[i] < E1) || (E[i] > E2)) {
//				printf("%c%d\n", prefix, k);
//				printf("pos: %d\n", i);
//				print_E(&(E[std::max(i - 5, 0)]), std::min(i, 5), 'b');
//				printf("E_err = %lf\n", E[i]);
////				print_S();
//				print_E(&(E[std::min(i + 5, N - 1)]), std::min(N - i, 5), 'a');
//				return 0;
//			}
//		return 1;
//	}
}

