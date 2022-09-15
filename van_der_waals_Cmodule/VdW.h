//
// Created by ypolyach on 10/27/21.
//

#ifndef VDW_VDW_H
#define VDW_VDW_H

#include <gsl/gsl_rng.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define gen_mode_ID_random 0
#define gen_mode_ID_cubic_lattice 1
#define N_gen_modes 2

#define dim 3
#define crd3D_loop for(int _ix = 0; _ix < dim; ++_ix)

namespace VdW
{
	extern gsl_rng *rng;
    extern int seed;
    extern int verbose_dafault;

	template <typename T> T sqr(T x) { return x * x; }
	template <typename T> void zero_array(T* v, long N, T v0=T(0)) { for(long i = 0; i < N; ++i) v[i] = v0; }
//	template <typename T> char sgn(T val) { return (T(0) < val) - (val < T(0));	}
	template <typename T> char sgn(T val) { return T(0) <= val ? 1 : -1;	}
	template <typename T> T powi(T x, int N) {
		T res = T(1);
		for(int i = 0; i < N; ++i) res *= x;
		return res;
	}
	template <typename T> T sum_array(T* v, unsigned long N) {
		T s = T(0);
		for(unsigned long i = 0; i < N; ++i) s += v[i];
		return s;
	}
	template <typename T> T max(T *v, unsigned long N) {
		T mx = v[0];
		for(unsigned long i = 0; i < N; ++i) if(mx < v[i]) mx = v[i];
		return  mx;
	}

	double md(double x, double L);
	void md_3D(double *X, double L);
	void add_to_3D(double *X, const double *Y);
	void add_to_3D(double *X, double a);
	void substract_from_3D(double *X, const double *Y);
	void mult_by_3D(double *X, double a);
	void get_random_3D_point(double *X, double L);
	void assign_to_3D(double *X, const double *Y);
	double dist2(const double *X1, const double *X2, double R, int m=0);
	double u_fnc(double r2, double lmd2);

	void print_E(const double *E, long Nt, char prefix=0, char suffix=0);
	void print_X(const double *X, char prefix=0);
	void print_state(const double *X, int N_atoms, char prefix=0);
//	int E_is_valid(const double *E, const double E1, const double E2, int N, int k=0, char prefix=0);
//	int state_is_valid(const int *s, int L, int k=0, char prefix=0);

	int run_bruteforce_C(double L, double Temp, double lmd, double dl, int N_atoms, long Nt_max, int gen_mode,
						 long *OP_arr_len, long *Nt, double **E, int **biggest_cluster_sizes,
						 int verbose);
	int run_state(double *X, double L, double Temp, double lmd2, double dl, int N_atoms,
				  double **E, int **biggest_cluster_sizes,
				  int *cluster_element_inds, int *cluster_sizes, int *is_checked, long *Nt, long *OP_arr_len,
				  int verbose, long Nt_max);
//	int get_init_states_C(int L, double Temp, double h, int N_init_states, int *init_states, int mode, int OP_thr_save_state,
//						  int interface_mode, int default_spin_state, int OP_A, int OP_B,
//						  double **E, int **M, int **biggest_cluster_size, int **h_A,
//						  long *Nt, long *OP_arr_len, int verbose);

	int init_rand_C(int my_seed);
	double comp_E(const double * X, double R, double lmd2, int N_atoms);
	bool X_fits_in(const double *X_new, const double *X_present, int N_atoms_present, double R, double r2_min=1);
	void generate_state(double *X, double L, int N_atoms, int gen_mode, int verbose);
	double flip_random_particle(double *X, double L, double R, double lmd2, double dl, double Temp, int N_atoms, int* i_atom);
	double get_dE_atom(double *X, double L, double R, double lmd2, int N_atoms, int i_atom, double *dX);
	double get_Ei(double *X, double R, double lmd2, int N_atoms, int i_atom, double *X_atom);

	void cluster_state_C(const double *X, double R, double lmd2, int N_atoms, int *cluster_element_inds, int *cluster_sizes, int *N_clusters, int *is_checked);
	void add_to_cluster(const double* X, double R, double lmd2, int N_atoms, int* is_checked, int* cluster, int* cluster_size, int i_atom, int cluster_label);
//	int is_infinite_cluster(const int* cluster, const int* cluster_size, int L, char *present_rows, char *present_columns);
	void uncheck_state(int *is_checked, int N_atoms);
	void clear_clusters(int *clusters, int *cluster_sizes, int *N_clusters);
}

//py::tuple get_init_states(int L, double Temp, double h, int N0, int M_0, int to_get_EM, std::optional<int> _verbose);
py::tuple run_bruteforce(double L, int N_atoms, double Temp, double lmd, double dl, long Nt_max,
						 std::optional<int> _to_remember_timeevol, std::optional<int> _verbose, int gen_mode);
//py::tuple cluster_state(py::array_t<int> state, int default_state, std::optional<int> _verbose);
py::int_ init_rand(int my_seed);
py::int_ set_verbose(int new_verbose);
py::int_ get_seed();

#endif //VDW_VDW_H
