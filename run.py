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

def run_many(L, N_atoms, Temp, lmd, dl, N_runs, Nt=-5, verbose=None, \
			to_get_timeevol=True, gen_mode=VdW.get_gen_mode_ID_cubic_lattice(), \
			to_plot_timeevol=False, timeevol_stride=-3000, \
			to_plot_k_distr=False, N_k_bins=None, mode='BF', \
			to_plot_individual_timeevol=False):
	old_seed = VdW.get_seed()
	if(to_get_timeevol is None):
		to_get_timeevol = ('BF' == mode)
	
	if(N_k_bins is None):
		N_k_bins = int(np.round(np.sqrt(N_runs) / 2) + 1)
	
	E_data = np.empty(N_runs)
	d_E_data = np.empty(N_runs)
	CS_data = np.empty(N_runs)
	d_CS_data = np.empty(N_runs)
	
	for i in range(N_runs):
		VdW.init_rand(i * 12345 + old_seed)
		if(mode == 'BF'):
			E_new, CS_new, timesteps, E_data[i], d_E_data[i], CS_data[i], d_CS_data[i], \
				dt, rho, stab_Nsteps, timeevol_stride = \
					proc_T(L, N_atoms, Temp, lmd, dl, Nt=Nt, \
							to_get_timeevol=to_get_timeevol, \
							gen_mode=gen_mode, \
							to_plot_timeevol=to_plot_individual_timeevol, \
							timeevol_stride=timeevol_stride)
			if(i == 0):
				N_E = len(E_new)
				E_time_data = np.empty((N_E, N_runs))
				CS_time_data = np.empty((N_E, N_runs))
			E_time_data[:, i] = E_new
			CS_time_data[:, i] = CS_new
	VdW.init_rand(old_seed)
	
	#dt = dl**2 / 2 / N_atoms
	#rho = N_atoms / (L**3)
	
	#timesteps = np.arange(N_E) * timeevol_stride
	E_mean, d_E_mean = get_average(E_data)
	CS_mean, d_CS_mean = get_average(CS_data)
	E_time, d_E_time = get_average(E_time_data, axis=1)
	CS_time, d_CS_time = get_average(CS_time_data, axis=1)
	# E_time = np.mean(E_time_data, axis=1)
	# d_E_time = np.std(E_time_data, axis=1) / np.sqrt(N_runs - 1)
	# CS_time = np.mean(CS_time_data, axis=1)
	# d_CS_time = np.std(CS_time_data, axis=1) / np.sqrt(N_runs - 1)
	
	if(to_plot_k_distr):
		plot_k_distr(E_data, N_k_bins, r'<E> / \varepsilon')
		plot_k_distr(CS_data, N_k_bins, r'CS')
	
	if(to_plot_timeevol):
		plot_E_CS(timesteps, E_time, CS_time, N_atoms, dt, Temp, lmd, dl, rho, stab_Nsteps, d_E_time=d_E_time, d_CS_time=d_CS_time)
		# fig_E, ax_E = my.get_fig(r'$t D_{dl} / \sigma^2$', r'$[E / \epsilon] / [N(N-1)/2]$', \
								# title=r'E(t); $T / \epsilon = ' + str(Temp) + \
										# r'$; $\lambda / \sigma = ' + str(lmd - 1) + \
										# r'$; $dl/\sigma = ' + str(dl) + \
										# r'$; $\rho \sigma^3 = ' + my.f2s(rho) + '$')
		# fig_CS, ax_CS = my.get_fig(r'$t D_{dl} / \sigma^2$', r'$CS / N$', \
								# title=r'CS(t); $T / \epsilon = ' + str(Temp) + \
										# r'$; $\lambda / \sigma = ' + str(lmd - 1) + \
										# r'$; $dl/\sigma = ' + str(dl) + \
										# r'$; $\rho \sigma^3 = ' + my.f2s(rho) + '$')
		
		# E_scale = N_atoms * (N_atoms - 1) / 2
		# ax_E.errorbar(timesteps * dt, E_time / E_scale, yerr = d_E_time / E_scale, \
						# fmt='.', label='data')
		# ax_E.plot([stab_Nsteps * dt] * 2, np.array([min(E_time), max(E_time)]) / E_scale, label='relax time')
		# ax_E.legend()
		
		# CS_scale = N_atoms
		# ax_CS.errorbar(timesteps * dt, CS_time / CS_scale, yerr = d_CS_time / CS_scale, \
					# fmt='.', label='data')
		# ax_CS.plot([stab_Nsteps * dt] * 2, np.array([min(CS_time), max(CS_time)]) / CS_scale, label='relax time')
		# ax_CS.legend()
	
	if(mode == 'BF'):
		return E_mean, d_E_mean, CS_mean, d_CS_mean

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

def get_rho(Lx, Ly, Lz, states, dz, N_atoms):
	Nt = int(len(states) / 3 / N_atoms)
	assert(Nt * N_atoms * 3 == len(states))
	
	states = states.reshape((Nt, N_atoms, 3))
	
	N_z_intervals = int(Lz / dz + 0.5)
	dz = Lz / N_z_intervals
	
	Z_edges = np.arange(N_z_intervals + 1) * dz
	Z_centers = (Z_edges[1:] + Z_edges[:-1]) / 2
	rho = np.empty((Nt, N_z_intervals))
	d_rho = np.empty((Nt, N_z_intervals))
	dV = Lx * Ly * dz
	for it in range(Nt):
		rho[it, :], _ = np.histogram(states[it, :, 2], bins=Z_edges)
		d_rho[it, :] = np.sqrt(rho[it, :] * (1 - rho[it, :] / N_atoms))
		
		rho[it, :] = rho[it, :] / dV
		d_rho[it, :] = d_rho[it, :] / dV
	
	return Z_edges, rho, d_rho

def plot_rho(fig_rho, ax_rho, t, rho, d_rho, Z_edges, N_atoms, dt, Temp, lmd, dl, stab_Nsteps):
	Nt = rho.shape[0]
	Z_centers = (Z_edges[1:] + Z_edges[:-1]) / 2
	Z_lens = Z_edges[1:] - Z_edges[:-1]
	
	lbl = r'$T / \epsilon = ' + str(Temp) + \
			r'$; $\lambda / \sigma = ' + str(lmd - 1) + \
			r'$; $dl/\sigma = ' + str(dl) + '$'
	
	plt.ion()
	rho_curve, = ax_rho.plot(Z_centers, rho[0, :], '.')
	plt.draw()
	plt.pause(0.0001)

	for it in range(1, Nt):
		rho_curve.set_ydata(rho[it, :])
		ax_rho.set_ylim(min(rho[it, :]), max(rho[it, :]))
		fig_rho.canvas.draw()
		fig_rho.canvas.flush_events()
		ax_rho.set_title(r'$\rho(t = ' + my.f2s(t[it], n=3) + ', z)$; ' + lbl)
		plt.draw()
		plt.pause(0.0001)
	
	# fig_rho2D_id=3
	# fig_rho2D, ax_rho2D = my.get_fig(r'$(t D_{dl}) / a^2$', r'$z / \sigma$', \
						# title=r'$rho(t, z) \sigma^3$;' + lbl, fig_num=fig_rho_id)
	
	
	#rho_im = ax_rho.imshow(rho, vmin = np.min(rho), vmax = np.max(rho),
     #            extent =[0, 1, 0, T * D / l**2], interpolation ='nearest', origin ='lower', aspect='auto')
	#plt.figure(fig_rho_id)
	#plt.colorbar(rho_im)
	
	
	#ax_rho.bar(Z_centers, rho, d_rho, yerr=d_rho, width=Z_lens, align='center')


def plot_E_CS(timesteps, E_time, CS_time, N_atoms, dt, Temp, lmd, dl, stab_Nsteps, d_E_time=None, d_CS_time=None):
	lbl = '$T / \epsilon = ' + str(Temp) + \
			r'$; $\lambda / \sigma = ' + str(lmd - 1) + \
			r'$; $dl/\sigma = ' + str(dl) + '$'
	fig_E, ax_E = my.get_fig(r'$(t D_{dl}) / (L^3 / N)^{2/3}$', r'$[E / \epsilon] / [N(N-1)/2]$', \
							title=r'E(t); ' + lbl)
	fig_CS, ax_CS = my.get_fig(r'$(t D_{dl}) / (L^3 / N)^{2/3}$', r'$CS / N$', \
							title=r'CS(t); ' + lbl)

	E_scale = N_atoms * (N_atoms - 1) / 2
	ax_E.errorbar(timesteps * dt, E_time / E_scale, yerr = None if(d_E_time is None) else (d_E_time / E_scale), \
					fmt='.', label='data')
	ax_E.plot([stab_Nsteps * dt] * 2, np.array([min(E_time), max(E_time)]) / E_scale, label='relax time')
	ax_E.legend()

	CS_scale = N_atoms
	ax_CS.errorbar(timesteps * dt, CS_time / CS_scale, yerr = None if(d_CS_time is None) else (d_CS_time / CS_scale), \
				fmt='.', label='data')
	ax_CS.plot([stab_Nsteps * dt] * 2, np.array([min(CS_time), max(CS_time)]) / CS_scale, label='relax time')
	ax_CS.legend()

def proc_T(Lx, Ly, Lz, N_atoms, Temp, lmd, dl, Nt=-5, verbose=None, \
			to_get_timeevol=True, gen_mode=VdW.get_gen_mode_ID_cubic_lattice(), \
			rho_init=None, rho_dz=None, \
			to_plot_timeevol=False, timeevol_stride=-3000):
	
	rho = N_atoms / (Lx * Ly * Lz)
	if(gen_mode == VdW.get_gen_mode_ID_cubic_lattice()):
		rho_init = rho
	a_lattice = rho_init ** (-1/3)
	dt = (dl**2 / 2 / N_atoms) / (a_lattice)**2
	stab_Nsteps = int(1 / dt)   # all atoms should move ~ (L/N_lin)
	if(Nt < 0):
		Nt = stab_Nsteps * (-Nt)
	Nt = int(Nt)
	if(to_get_timeevol):
		if(timeevol_stride < 0):
			timeevol_stride = np.int_(- Nt / timeevol_stride)
	
	(states, E, CS) = VdW.run_bruteforce(Lx, Ly, Lz, N_atoms, Temp, lmd, dl, Nt, \
								to_remember_timeevol=to_get_timeevol, \
								verbose=verbose, \
								timestep_to_save_E=timeevol_stride, \
								gen_mode=gen_mode, gen_density=rho_init)
	
	# input(str(E.shape))
	
	N_E = len(E)
	timesteps = np.arange(N_E) * timeevol_stride
	stab_inds = timesteps > stab_Nsteps
	#E_mean = np.mean(E[stab_inds])
	#d_E_mean = np.std(E[stab_inds])
	#CS_mean = np.mean(CS[stab_inds])
	#d_CS_mean = np.std(CS[stab_inds])

	N_last = 100
	E_mean = np.mean(E[-N_last : ])
	d_E_mean = np.std(E[-N_last : ])
	CS_mean = np.mean(CS[-N_last : ])
	d_CS_mean = np.std(CS[-N_last : ])
	
	if(to_get_timeevol):
		Z_edges, rho, d_rho = get_rho(Lx, Ly, Lz, states, rho_dz, N_atoms)
		
		if(to_plot_timeevol):
			fig_rho, ax_rho = my.get_fig(r'$z$', r'$\rho$')
			
			plot_rho(fig_rho, ax_rho, timesteps * dt, rho, d_rho, Z_edges, N_atoms, dt, Temp, lmd, dl, stab_Nsteps)
			
			plot_E_CS(timesteps, E, CS, N_atoms, dt, Temp, lmd, dl, stab_Nsteps)
	
	return E, CS, timesteps, E_mean, d_E_mean, CS_mean, d_CS_mean, dt, rho, stab_Nsteps, timeevol_stride


def main():
	# BF: python run.py -Lx 5 -Ly 5 -Lz 15 -rho_init 1.0 -N_atoms 125 -Temp 0.5 -lmd 1.0 -dl 0.1 -run_mode BF -Nt_max 1.0
	[Lx, Ly, Lz, rho_init, rho_dz, N_atoms, Temp, lmd, dl, run_mode, gen_mode, Nt_max, to_get_timeevol, verbose, timeevol_stride, to_plot_timeevol, my_seed, N_runs], _ = \
		my.parse_args(sys.argv,            ['-Lx',  '-Ly',  '-Lz',  '-rho_init',  '-rho_dz',  '-N_atoms', '-Temp', '-lmd',  '-dl', '-run_mode',                                '-gen_mode',  '-Nt_max',  '-to_get_timeevol', '-verbose',  '-timeevol_stride',  '-to_plot_timeevol', '-my_seed', '-N_runs'], \
					  possible_arg_numbers=[  [1],    [1],    [1],          [1],        [1],         [1],     [1],    [1],    [1],         [1],                                     [0, 1],     [0, 1],              [0, 1],     [0, 1],              [0, 1],               [0, 1],     [0, 1],     [0, 1]], \
					  default_values=      [ None,   None,   None,         None,       None,        None,    None,   None,   None,        None,  [str(VdW.get_gen_mode_ID_solidUvacuum())],     ['-5'],   [my.yes_flags[0]],      ['1'],           ['-300'],    [my.yes_flags[0]],      ['0'],      ['5']])
	
	Lx = float(Lx)
	Ly = float(Ly)
	Lz = float(Lz)
	rho_init = float(rho_init)
	rho_dz = float(rho_dz)
	N_atoms = int(N_atoms)
	Temp = float(Temp)
	lmd = float(lmd)
	dl = float(dl)
	gen_mode = int(gen_mode[0])
	Nt_max = float(Nt_max[0])
	to_get_timeevol = (to_get_timeevol[0] in my.yes_flags)
	verbose = int(verbose[0])
	timeevol_stride = int(timeevol_stride[0])
	to_plot_timeevol = (to_plot_timeevol[0] in my.yes_flags)
	my_seed = int(my_seed[0])
	N_runs = int(N_runs[0])
	
	print('L=(%lf,%lf,%lf), rho_init=%lf, rho_dz=%lf, N_atoms=%d, Temp=%lf, lmd=%lf, dl=%lf, gen_mode=%d, Nt_max=%lf, to_get_timeevol:%s, verbose:%d, stride:%d, to_plot_timeevol:%s, my_seed:%d' % \
		(Lx, Ly, Lz, rho_init, rho_dz, N_atoms, Temp, lmd, dl, gen_mode, Nt_max, str(to_get_timeevol), verbose, timeevol_stride, str(to_plot_timeevol), my_seed))
	
	VdW.init_rand(my_seed)
	VdW.set_verbose(verbose)
	
	if(run_mode == 'BF'):
		proc_T(Lx, Ly, Lz, N_atoms, Temp, lmd, dl, Nt=Nt_max, rho_init=rho_init, rho_dz=rho_dz, \
			to_get_timeevol=to_get_timeevol, gen_mode=gen_mode, \
			to_plot_timeevol=to_plot_timeevol, timeevol_stride=timeevol_stride)
		
	elif(run_mode == 'BF_many'):
		run_many(L, N_atoms, Temp, lmd, dl, N_runs, Nt=Nt_max, \
			to_get_timeevol=to_get_timeevol, gen_mode=gen_mode, \
			to_plot_timeevol=to_plot_timeevol, timeevol_stride=timeevol_stride, \
			to_plot_k_distr=False, mode='BF')
		
	else:
		print('ERROR: run_mode=%s is not supported' % (run_mode))
	
	plt.show()

if(__name__ == "__main__"):
	main()

