import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import sys
import scipy.interpolate

import mylib as my

# ========================== recompile ==================
if(__name__ == "__main__"):
	to_recompile = True
	if(to_recompile):
		path_to_so = 'van_der_waals_Cmodule/cmake-build-debug'
		path_to_so = 'van_der_waals_Cmodule/cmake-build-release'
		N_dirs_down = path_to_so.count('/') + 1
		path_back = '/'.join(['..'] * N_dirs_down)

		os.chdir(path_to_so)
		my.run_it('make VdW.so')
		#my.run_it('cmake  --build cmake-build-debug --target VdW.so')
		os.chdir(path_back)
		my.run_it('mv %s/VdW.so.cpython-38-x86_64-linux-gnu.so ./VdW.so' % (path_to_so))
		print('recompiled VdW')

	import VdW
	#exit()

# ========================== functions ==================

# this should match VdW.h definition
gen_mode_ID_cubic_lattice = 1

def get_average(x_data, axis=0, mode='lin'):
	N_runs = x_data.shape[axis]
	if(mode == 'log'):
		x = np.exp(np.mean(np.log(x_data), axis=axis))
		d_x = x * np.std(np.log(x_data), axis=axis) / np.sqrt(N_runs - 1)   # dx = x * d_lnx = x * dx/x
	else:
		x = np.mean(x_data, axis=axis)
		d_x = np.std(x_data, axis=axis) / np.sqrt(N_runs - 1)
	return x, d_x

def plot_k_distr(k_data, N_bins, lbl, units=None):
	N_runs = len(k_data)
	k_hist, k_edges = np.histogram(k_data, bins=N_bins)
	d_k_hist = np.sqrt(k_hist * (1 - k_hist / N_runs))
	k_centers = (k_edges[1:] + k_edges[:-1]) / 2
	k_lens = k_edges[1:] - k_edges[:-1]
	
	units_inv_str = ('' if(units is None) else (' $(' + units + ')^{-1}$'))
	units_str = ('' if(units is None) else (' $(' + units + ')$'))
	fig, ax = my.get_fig(r'$%s$%s' % (lbl, units_inv_str), r'$\rho$%s' % (units_str), r'$\rho(%s)$' % (lbl))
	
	ax.bar(k_centers, k_hist / k_lens, yerr=d_k_hist / k_lens, width=k_lens, align='center')

def proc_T(L, N_atoms, Temp, lmd, dl, Nt=-5, verbose=None, to_get_timeevol=True, init_gen_mode=gen_mode_ID_cubic_lattice, \
			to_plot_timeevol=False, timeevol_stride=-3000):
	
	dt = dl**2 / 2
	rho = N_atoms / (L**3)
	N_lin = int(np.ceil(N_atoms ** (1/3)) + 0.1)
	stab_Nsteps = int(N_atoms * (L / N_lin / dl)**2) * 2   # all atoms should move ~ (L/N_lin)
	if(Nt < 0):
		Nt = stab_Nsteps * (-Nt)
	
	(E, CS) = VdW.run_bruteforce(L, N_atoms, Temp, lmd, dl, Nt, to_remember_timeevol=to_get_timeevol, verbose=verbose)
	
	# input(str(E.shape))
	
	timesteps = np.arange(Nt)
	stab_inds = timesteps > stab_Nsteps
	E_mean = np.mean(E[stab_inds])
	d_E_mean = np.std(E[stab_inds])
	CS_mean = np.mean(CS[stab_inds])
	d_CS_mean = np.std(CS[stab_inds])
	
	if(to_get_timeevol):
		if(timeevol_stride < 0):
			timeevol_stride = np.int_(- Nt / timeevol_stride)
		if(to_plot_timeevol):
			fig_E, ax_E = my.get_fig(r'$t D_{dl} / \sigma^2$', r'$E / \epsilon / (N(N-1)/2)$', \
									title=r'E(t); $T / \epsilon = ' + str(Temp) + \
											r'$; $\lambda / \sigma = ' + str(lmd - 1) + \
											r'$; $dl/\sigma = ' + str(dl) + \
											r'$; $\rho \sigma^3 = ' + my.f2s(rho) + '$')
			fig_CS, ax_CS = my.get_fig(r'$t D_{dl} / \sigma^2$', r'$CS$', \
									title=r'E(t); $T / \epsilon = ' + str(Temp) + \
											r'$; $\lambda / \sigma = ' + str(lmd - 1) + \
											r'$; $dl/\sigma = ' + str(dl) + \
											r'$; $\rho \sigma^3 = ' + my.f2s(rho) + '$')
			
			ax_E.plot(timesteps[::timeevol_stride] * dt, E[::timeevol_stride] / (N_atoms * (N_atoms - 1) / 2), '.', label='data')
			ax_E.plot([stab_Nsteps * dt] * 2, np.array([min(E), max(E)]) / (N_atoms * (N_atoms - 1) / 2), label='relax time')
			ax_E.legend()
			
			ax_CS.plot(timesteps[::timeevol_stride] * dt, CS[::timeevol_stride], '.', label='data')
			ax_CS.plot([stab_Nsteps * dt] * 2, np.array([min(CS), max(CS)]), label='relax time')
			ax_CS.legend()
		# return F, d_F, hist_centers, hist_lens, rho_interp1d, d_rho_interp1d, k_bc_AB, k_bc_BA, k_AB, d_k_AB, k_BA, d_k_BA, OP_avg, d_OP_avg, E_avg#, C


def run_many(L, Temp, h, N_runs, interface_mode, def_spin_state, \
			OP_A=None, OP_B=None, N_spins_up_init=None, \
			OP_interfaces_AB=None, OP_interfaces_BA=None, \
			OP_sample_BF_A_to=None, OP_sample_BF_B_to=None, \
			OP_match_BF_A_to=None, OP_match_BF_B_to=None, \
			Nt_per_BF_run=None, verbose=None, to_plot_timeevol=False, \
			timeevol_stride=-3000, to_plot_k_distr=False, N_k_bins=10, \
			mode='BF', N_init_states_AB=None, N_init_states_BA=None, \
			init_gen_mode=-2, to_plot_committer=None, to_get_timeevol=None):
	L2 = L**2
	old_seed = VdW.get_seed()
	if(to_get_timeevol is None):
		to_get_timeevol = ('BF' == mode)
	
	if('BF' in mode):
		assert(Nt_per_BF_run is not None), 'ERROR: BF mode but no Nt_per_run provided'
		if(OP_A is None):
			assert(len(OP_interfaces_AB) > 0), 'ERROR: nor OP_A neither OP_interfaces_AB were provided for run_many(BF)'
			OP_A = OP_interfaces_AB[0]
		if('AB' not in mode):
			if(OP_B is None):
				assert(len(OP_interfaces_BA) > 0), 'ERROR: nor OP_B neither OP_interfaces_BA were provided for run_many(BF)'
				OP_B = flip_OP(OP_interfaces_BA[0], L2, interface_mode)
		
	elif('FFS' in mode):
		assert(N_init_states_AB is not None), 'ERROR: FFS_AB mode but no "N_init_states_AB" provided'
		assert(OP_interfaces_AB is not None), 'ERROR: FFS_AB mode but no "OP_interfaces_AB" provided'
		
		N_OP_interfaces_AB = len(OP_interfaces_AB)
		OP_AB = OP_interfaces_AB / OP_scale[interface_mode]
		if(mode == 'FFS'):
			assert(N_init_states_BA is not None), 'ERROR: FFS mode but no "N_init_states_BA" provided'
			assert(OP_interfaces_BA is not None), 'ERROR: FFS mode but no "OP_interfaces_BA" provided'
			
			N_OP_interfaces_BA = len(OP_interfaces_BA)
			OP_BA = flip_OP(OP_interfaces_BA, L2, interface_mode) / OP_scale[interface_mode]
			OP = np.sort(np.unique(np.concatenate((OP_AB, OP_BA))))
		elif(mode == 'FFS_AB'):
			OP = np.copy(OP_AB)
	
	ln_k_AB_data = np.empty(N_runs)
	ln_k_BA_data = np.empty(N_runs) 
	if(to_get_timeevol):
		rho_fncs = [[]] * N_runs
		d_rho_fncs = [[]] * N_runs
	if(mode == 'BF'):
		ln_k_bc_AB_data = np.empty(N_runs)
		ln_k_bc_BA_data = np.empty(N_runs) 
	elif('FFS' in mode):
		flux0_AB_data = np.empty(N_runs) 
		OP0_AB_data = np.empty(N_runs) 
		probs_AB_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		PB_AB_data = np.empty((N_OP_interfaces_AB, N_runs))
		d_PB_AB_data = np.empty((N_OP_interfaces_AB, N_runs))
		PB_sigmoid_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		d_PB_sigmoid_data = np.empty((N_OP_interfaces_AB - 1, N_runs))
		PB_linfit_data = np.empty((2, N_runs))
		PB_linfit_inds_data = np.empty((N_OP_interfaces_AB - 1, N_runs), dtype=bool)
		
		if(mode == 'FFS'):
			flux0_BA_data = np.empty(N_runs)
			OP0_BA_data = np.empty(N_runs)
			probs_BA_data = np.empty((N_OP_interfaces_BA - 1, N_runs))
			PA_BA_data = np.empty((N_OP_interfaces_BA, N_runs))
			d_PA_BA_data = np.empty((N_OP_interfaces_BA, N_runs))
			PA_sigmoid_data = np.empty((N_OP_interfaces_BA - 1, N_runs))
			d_PA_sigmoid_data = np.empty((N_OP_interfaces_BA - 1, N_runs))
			PA_linfit_data = np.empty((2, N_runs))
			PA_linfit_inds_data = np.empty((N_OP_interfaces_BA - 1, N_runs), dtype=bool)
	
	for i in range(N_runs):
		VdW.init_rand(i + old_seed)
		if(mode == 'BF'):
			F_new, d_F_new, OP_hist_centers_new, OP_hist_lens_new, \
				rho_fncs[i], d_rho_fncs[i], k_bc_AB_BF, k_bc_BA_BF, k_AB_BF, _, k_BA_BF, _, _, _, _ = \
					proc_T(L, Temp, h, Nt_per_BF_run, interface_mode, def_spin_state, \
							OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
							verbose=verbose, to_plot_timeevol=to_plot_timeevol, \
							to_plot_F=False, to_plot_correlations=False, to_plot_ETS=False, \
							timeevol_stride=timeevol_stride, to_estimate_k=True, 
							to_get_timeevol=True)
			
			ln_k_bc_AB_data[i] = np.log(k_bc_AB_BF * 1)   # proc_T returns 'k', not 'log(k)'; *1 for units [1/step]
			ln_k_bc_BA_data[i] = np.log(k_bc_BA_BF * 1)
			ln_k_AB_data[i] = np.log(k_AB_BF * 1)
			ln_k_BA_data[i] = np.log(k_BA_BF * 1)
		elif(mode  == 'BF_AB'):
			k_AB_BFcount, _ = \
					proc_T(L, Temp, h, Nt_per_BF_run, interface_mode, def_spin_state, \
							OP_A=OP_A, OP_B=OP_B, N_spins_up_init=N_spins_up_init, \
							verbose=verbose, to_plot_timeevol=to_plot_timeevol, \
							to_plot_F=False, to_plot_correlations=False, to_plot_ETS=False, \
							timeevol_stride=timeevol_stride, to_estimate_k=False, 
							to_get_timeevol=True)
			
			ln_k_AB_data[i] = np.log(k_AB_BFcount * 1)
		elif(mode == 'FFS_AB'):
			probs_AB_data[:, i], _, ln_k_AB_data[i], _, flux0_AB_data[i], _, rho_new, d_rho_new, OP_hist_centers_new, OP_hist_lens_new, \
			PB_AB_data[:, i], d_PB_AB_data[:, i], PB_sigmoid_data[:, i], d_PB_sigmoid_data[:, i], PB_linfit_data[:, i], PB_linfit_inds_data[:, i], OP0_AB_data[i] = \
				proc_FFS_AB(L, Temp, h, N_init_states_AB, OP_interfaces_AB, interface_mode, def_spin_state, \
							OP_sample_BF_to=OP_sample_BF_A_to, OP_match_BF_to=OP_match_BF_A_to, init_gen_mode=init_gen_mode, to_get_timeevol=to_get_timeevol)
			if(to_get_timeevol):
				rho_fncs[i] = scipy.interpolate.interp1d(OP_hist_centers_new, rho_new, fill_value='extrapolate')
				d_rho_fncs[i] = scipy.interpolate.interp1d(OP_hist_centers_new, d_rho_new, fill_value='extrapolate')
				rho_AB = rho_fncs[i](OP_hist_centers_new)
				d_rho_AB = d_rho_fncs[i](OP_hist_centers_new)
				F_new = -Temp * np.log(rho_AB * OP_hist_lens_new)
				d_F_new = Temp * d_rho_AB / rho_AB
				F_new = F_new - F_new[0]

		elif(mode == 'FFS'):
			F_new, d_F_new, OP_hist_centers_new, OP_hist_lens_new, rho_fncs[i], d_rho_fncs[i], probs_AB_data[:, i], _, \
				ln_k_AB_data[i], _, flux0_AB_data[i], _, probs_BA_data[:, i], _, ln_k_BA_data[i], _, flux0_BA_data[i], _, \
				PB_AB_data[:, i], d_PB_AB_data[:, i], PA_BA_data[:, i], d_PA_BA_data[:, i], \
				PB_sigmoid_data[:, i], d_PB_sigmoid_data[:, i], PA_sigmoid_data[:, i], d_PA_sigmoid_data[:, i], \
				PB_linfit_data[:, i], PA_linfit_data[:, i], \
				PB_linfit_inds_data[:, i], PA_linfit_inds_data[:, i], \
				OP0_AB_data[i], OP0_BA_data[i] = \
					proc_FFS(L, Temp, h, \
						N_init_states_AB, N_init_states_BA, \
						OP_interfaces_AB, OP_interfaces_BA, \
						OP_sample_BF_A_to, OP_sample_BF_B_to, \
						OP_match_BF_A_to, OP_match_BF_B_to, \
						interface_mode, def_spin_state, \
						init_gen_mode=init_gen_mode, \
						to_get_timeevol=to_get_timeevol)
			
		print(r'%s %lf %%' % (mode, (i+1) / (N_runs) * 100))
		
		if(to_get_timeevol):
			if(i == 0):
				OP_hist_centers = OP_hist_centers_new
				OP_hist_lens = OP_hist_lens_new
				N_OP_hist_points = len(OP_hist_centers)
				F_data = np.empty((N_OP_hist_points, N_runs))
				d_F_data = np.empty((N_OP_hist_points, N_runs))
			else:
				def check_OP_centers_match(c, i, c_new, arr_name, scl=1, err_thr=1e-6):
					rel_err = abs(c_new - c) / scl
					miss_inds = rel_err > err_thr
					assert(np.all(~miss_inds)), r'ERROR: %s from i=0 and i=%d does not match:\nold: %s\nnew: %s\nerr: %s' % (arr_name, i, str(c[miss_inds]), str(c_new[miss_inds]), str(rel_err[miss_inds]))
				
				check_OP_centers_match(OP_hist_centers, i, OP_hist_centers_new, 'OP_hist_centers', max(abs(OP_hist_centers[1:] - OP_hist_centers[:-1])))
				check_OP_centers_match(OP_hist_lens, i, OP_hist_lens_new, 'OP_hist_lens', max(OP_hist_lens))
			F_data[:, i] = F_new
			d_F_data[:, i] = d_F_new
	
	ln_k_AB, d_ln_k_AB = get_average(ln_k_AB_data)
	
	if(to_get_timeevol):
		F, d_F = get_average(F_data, axis=1)
	else:
		F = None
		d_F = None
		OP_hist_centers = None
		OP_hist_lens = None
	
	if('AB' not in mode):
		ln_k_BA, d_ln_k_BA = get_average(ln_k_BA_data)

	if(mode == 'BF'):
		ln_k_bc_AB, d_ln_k_bc_AB = get_average(ln_k_bc_AB_data)
		ln_k_bc_BA, d_ln_k_bc_BA = get_average(ln_k_bc_BA_data)
		
	elif('FFS' in mode):
		flux0_AB, d_flux0_AB = get_average(flux0_AB_data, mode='log')
		OP0_AB, d_OP0_AB = get_average(OP0_AB_data)
		probs_AB, d_probs_AB = get_average(probs_AB_data, mode='log', axis=1)
		PB_AB, d_PB_AB = get_average(PB_AB_data, mode='log', axis=1)

		PB_sigmoid, d_PB_sigmoid, linfit_AB, linfit_AB_inds, m0_AB_fit = \
			get_sigmoid_fit(PB_AB[:-1], d_PB_AB[:-1], OP_AB[:-1], h, table_data.Nc_reference_data[str(Temp)])

		if('AB' not in mode):
			flux0_BA, d_flux0_BA = get_average(flux0_BA_data, mode='log')
			OP0_BA, d_OP0_BA = get_average(OP0_BA_data)
			probs_BA, d_probs_BA = get_average(probs_BA_data, mode='log', axis=1)
			PA_BA, d_PA_BA = get_average(PA_BA_data, mode='log', axis=1)
		
			PA_sigmoid, d_PA_sigmoid, linfit_BA, linfit_BA_inds, m0_BA_fit = \
				get_sigmoid_fit(PA_BA[1:], d_PA_BA[1:], OP_BA[1:], h, table_data.Nc_reference_data[str(Temp)])

		# =========================== 4-param sigmoid =======================
		# if(interface_mode == 'M'):
			# tht0_AB = [0, -0.15]
			# tht0_BA = [tht0_AB[0], - tht0_AB[1]]
		# elif(interface_mode == 'CS'):
			# tht0_AB = [60, 10]
			# tht0_BA = [L2 - tht0_AB[0], - tht0_AB[1]]
		# PB_opt, PB_opt_fnc, PB_opt_fnc_inv = PB_fit_optimize(PB_AB[:-1], d_PB_AB[:-1], OP_AB[:-1], OP_AB[0], OP_AB[-1], PB_AB[0], PB_AB[-1], tht0=tht0_AB)
		# PA_opt, PA_opt_fnc, PA_opt_fnc_inv = PB_fit_optimize(PA_BA[1:], d_PA_BA[1:], OP_BA[1:], OP_BA[0], OP_BA[-1], PA_BA[0], PA_BA[-1], tht0=tht0_BA)
		# OP0_AB = PB_opt.x[0]
		# OP0_BA = PA_opt.x[0]
		# PB_fit_a, PB_fit_b = get_intermediate_vals(PB_opt.x[0], PB_opt.x[1], OP_AB[0], OP_AB[-1], PB_AB[0], PB_AB[-1])
		# PA_fit_a, PA_fit_b = get_intermediate_vals(PA_opt.x[0], PA_opt.x[1], OP_BA[0], OP_BA[-1], PA_BA[0], PA_BA[-1])

		#PB_sigmoid = -np.log(PB_fit_b / (PB_AB[:-1] - PB_fit_a) - 1)
		#d_PB_sigmoid = d_PB_AB[:-1] * PB_fit_b / ((PB_AB[:-1] - PB_fit_a) * (PB_fit_b - PB_AB[:-1] + PB_fit_a))
		#PA_sigmoid = -np.log(PA_fit_b / (PA_BA[:-1] - PA_fit_a) - 1)
		#d_PA_sigmoid = d_PA_BA[1:] * PA_fit_b / ((PA_BA[1:] - PA_fit_a) * (PA_fit_b - PA_BA[1:] + PA_fit_a))
		
	if(to_plot_k_distr):
		plot_k_distr(np.exp(ln_k_AB_data) / 1, N_k_bins, 'k_{AB}', units='step')
		plot_k_distr(ln_k_AB_data, N_k_bins, r'\ln(k_{AB} \cdot 1step)')
		if('AB' not in mode):
			plot_k_distr(np.exp(ln_k_BA_data) / 1, N_k_bins, 'k_{BA}', units='step')
			plot_k_distr(ln_k_BA_data, N_k_bins, r'\ln(k_{BA} \cdot 1step)')
		if(mode == 'BF'):
			plot_k_distr(np.exp(ln_k_bc_AB_data) / 1, N_k_bins, r'k_{bc, AB}', units='step')
			plot_k_distr(ln_k_bc_AB_data, N_k_bins, r'\ln(k_{bc, AB} \cdot 1step)')
			plot_k_distr(np.exp(ln_k_bc_BA_data) / 1, N_k_bins, r'k_{bc, BA}', units='step')
			plot_k_distr(ln_k_bc_BA_data, N_k_bins, r'\ln(k_{bc, BA} \cdot 1step)')
	
	if(to_plot_committer is None):
		to_plot_committer = ('FFS' in mode)
	else:
		assert(not (('BF' in mode) and to_plot_committer)), 'ERROR: BF mode used but commiter requested'
	
	if(to_plot_committer):
		fig_PB_log, ax_PB_log = my.get_fig(title[interface_mode], r'$P_B(' + feature_label[interface_mode] + ') = P(i|0)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h), yscl='log')
		fig_PB, ax_PB = my.get_fig(title[interface_mode], r'$P_B(' + feature_label[interface_mode] + ') = P(i|0)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		fig_PB_sigmoid, ax_PB_sigmoid = my.get_fig(title[interface_mode], r'$-\ln(1/P_B - 1)$', title=r'$P_B(' + feature_label[interface_mode] + ')$; T/J = ' + str(Temp) + '; h/J = ' + str(h))
		
		mark_PB_plot(ax_PB, ax_PB_log, ax_PB_sigmoid, OP, PB_AB, PB_sigmoid, interface_mode)
		# for i in range(N_runs):
			# plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, h, 'P_{B' + str(i) + '}', feature_label[interface_mode], OP_AB[:-1], PB_AB_data[:-1, i], d_PB=d_PB_AB_data[:-1, i], \
						# PB_sgm=PB_sigmoid_data[:, i], d_PB_sgm=d_PB_sigmoid_data[:, i], \
						# linfit=PB_linfit_data[:, i], linfit_inds=PB_linfit_inds_data[:, i], \
						# OP0=OP0_AB_data[i], d_OP0=None, \
						# clr=my.get_my_color(-i-1))
			# plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, h, 'P_{A' + str(i) + '}', feature_label[interface_mode], OP_BA[1:], PA_BA_data[1:, i], d_PB=d_PA_BA_data[1:, i], \
						# PB_sgm=PA_sigmoid_data[:, i], d_PB_sgm=d_PA_sigmoid_data[:, i], \
						# linfit=PA_linfit_data[:, i], linfit_inds=PA_linfit_inds_data[:, i], \
						# OP0=OP0_BA_data[i], d_OP0=None, \
						# clr=my.get_my_color(i+1))
		plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, h, 'P_B', feature_label[interface_mode], \
					OP_AB[:-1], PB_AB[:-1], d_PB=d_PB_AB[:-1], \
					PB_sgm=PB_sigmoid, d_PB_sgm=d_PB_sigmoid, \
					linfit=linfit_AB, linfit_inds=linfit_AB_inds, \
					OP0=OP0_AB, d_OP0=d_OP0_AB, \
					clr=my.get_my_color(1))
					
		if('AB' not in mode):
			plot_PB_AB(ax_PB, ax_PB_log, ax_PB_sigmoid, h, 'P_A', feature_label[interface_mode], \
						OP_BA[1:], PA_BA[1:], d_PB=d_PA_BA[1:], \
						PB_sgm=PA_sigmoid, d_PB_sgm=d_PA_sigmoid, \
						linfit=linfit_BA, linfit_inds=linfit_BA_inds, \
						OP0=OP0_BA, d_OP0=d_OP0_BA, \
						clr=my.get_my_color(2))
		
		ax_PB_log.legend()
		ax_PB.legend()
		ax_PB_sigmoid.legend()
	
	VdW.init_rand(old_seed)
	
	if(mode == 'FFS'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, flux0_AB, d_flux0_AB, flux0_BA, d_flux0_BA, probs_AB, d_probs_AB, probs_BA, d_probs_BA, OP0_AB, d_OP0_AB, OP0_BA, d_OP0_BA
	elif(mode == 'FFS_AB'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, flux0_AB, d_flux0_AB, probs_AB, d_probs_AB, OP0_AB, d_OP0_AB
	elif(mode == 'BF'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB, ln_k_BA, d_ln_k_BA, ln_k_bc_AB, d_ln_k_bc_AB, ln_k_bc_BA, d_ln_k_bc_BA
	elif(mode == 'BF_AB'):
		return F, d_F, OP_hist_centers, OP_hist_lens, ln_k_AB, d_ln_k_AB

def main():
	# TODO: remove outdated inputs
	[L, N_atoms, Temp, lmd, dl, run_mode, gen_mode, Nt_max, to_get_timeevol, verbose, timeevol_stride, to_plot_timeevol, my_seed], _ = \
		my.parse_args(sys.argv,            ['-L',  '-N_atoms', '-Temp', '-lmd',  '-dl', '-run_mode',                       '-gen_mode',  '-Nt_max',  '-to_get_timeevol', '-verbose',  '-timeevol_stride',  '-to_plot_timeevol', '-my_seed'], \
					  possible_arg_numbers=[ [1],         [1],     [1],    [1],    [1],         [1],                            [0, 1],     [0, 1],              [0, 1],     [0, 1],              [0, 1],               [0, 1],     [0, 1]], \
					  default_values=      [None,        None,    None,   None,   None,        None,  [str(gen_mode_ID_cubic_lattice)],     ['-5'],   [my.yes_flags[0]],      ['1'],           ['-3000'],    [my.yes_flags[0]],       ['0']])
	
	L = float(L)
	N_atoms = int(N_atoms)
	Temp = float(Temp)
	lmd = float(lmd)
	dl = float(dl)
	gen_mode = int(gen_mode[0])
	Nt_max = int(Nt_max[0])
	to_get_timeevol = (to_get_timeevol[0] in my.yes_flags)
	verbose = int(verbose[0])
	timeevol_stride = int(timeevol_stride[0])
	to_plot_timeevol = (to_plot_timeevol[0] in my.yes_flags)
	my_seed = int(my_seed[0])
	
	print('L=%lf, N_atoms=%d, Temp=%lf, lmd=%lf, dl=%lf, gen_mode=%d, Nt_max=%d, to_get_timeevol:%s, verbose:%d, stride:%d, to_plot_timeevol:%s, my_seed:%d' % \
		(L, N_atoms, Temp, lmd, dl, gen_mode, Nt_max, str(to_get_timeevol), verbose, timeevol_stride, str(to_plot_timeevol), my_seed))
	
	VdW.init_rand(my_seed)
	VdW.set_verbose(verbose)
	
	if(run_mode == 'BF'):
		proc_T(L, N_atoms, Temp, lmd, dl, Nt=Nt_max, to_get_timeevol=to_get_timeevol, init_gen_mode=gen_mode, \
			to_plot_timeevol=to_plot_timeevol, timeevol_stride=timeevol_stride)
	
	elif(run_mode == 'BF_many'):
		run_many(L, Temp, h[0], N_runs, interface_mode, def_spin_state, Nt_per_BF_run=Nt, \
				OP_A=OP_0[0], OP_B=OP_max[0], N_spins_up_init=N_spins_up_init, \
				to_plot_k_distr=True, N_k_bins=N_k_bins, \
				mode='BF')
	
	else:
		print('ERROR: run_mode=%s is not supported' % (run_mode))
	
	plt.show()

if(__name__ == "__main__"):
	main()

