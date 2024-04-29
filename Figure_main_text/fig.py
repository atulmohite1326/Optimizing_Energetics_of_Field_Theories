import numpy as np
from pylab import *
from matplotlib.ticker import  *
from matplotlib.pyplot import  *
from numpy import *
from matplotlib import cm
import scipy
from scipy import ndimage
from PIL import Image
import pickle
#from pdf2image import convert_from_path

space         = 0.07
nb_lines      = 1
fig_width_pt  = 246.
inches_per_pt = 1./72.
golden_mean   = .66
fig_width     = fig_width_pt*inches_per_pt
fig_height    = (fig_width*golden_mean)+space
fig_size      = [1.5*fig_width, fig_height]
params = {'legend.fontsize': 4,
	        'text.latex.preamble': [r"\usepackage{amstext}", r"\usepackage{amsmath}", r"\usepackage{amssymb}", r"\usepackage{stmaryrd}"],
          'axes.linewidth': 5e-1,
          'axes.labelsize': 4.5,
          #'text.fontsize': 4,
          'xtick.labelsize': 5,
          'ytick.labelsize': 5,
          'text.usetex': True,
          'figure.figsize': fig_size}
rcParams.update(params)

l01 = 4.5e-2
b01 = .56
w01 = .29
h01 = .42

l02 = l01
b02 = 7e-2
w02 = w01
h02 = h01

l11 = .385
b11 = b01
w11 = w01
h11 = h01

l12 = l11
b12 = b02
w12 = w11
h12 = h11

l21 = .705
b21 = b11
w21 = w11
h21 = h11

l22 = l21
b22 = b12
w22 = w11
h22 = h11

l31 = 1.09
b31 = b11
w31 = w11
h31 = h11

l32 = l31
b32 = b12
w32 = w11
h32 = h11

# =======================
#  Plot
# =======================

# --------
#  Data
# --------

data = loadtxt('/home/atul/PhD_thesis/Paper_write_OMTP_mean_var/Figure2 (another copy)/correct_explicit_final_1/100 N = 64, T = 0.01, eq_time = 100.0, a = 1.0, b = 1.0, k = 0.5, dt = 0.0005/output.txt')
time_a = data[:,2]
q_numerical_a = (data[:,3] + data[:,5])/10
analytical_delta_f = data[:,8]

data = loadtxt('/home/atul/PhD_thesis/Paper_write_OMTP_mean_var/Figure2 (another copy)/correct_explicit_final_1/100 N = 64, T = 0.1, eq_time = 100.0, a = 1.0, b = 1.0, k = 0.5, dt = 0.0005/output.txt')
time_a = data[:,2]
q_numerical_a_1 = (data[:,3] + data[:,5])/100

data = loadtxt('/home/atul/PhD_thesis/Paper_write_OMTP_mean_var/Figure2 (another copy)/correct_explicit_final_1/100 N = 64, T = 0.001, eq_time = 100.0, a = 1.0, b = 1.0, k = 0.5, dt = 0.0005/output.txt')
time_a = data[:,2]
q_numerical_a_2 = (data[:,3] + data[:,5])

data = loadtxt('/home/atul/PhD_thesis/Paper_write_OMTP_mean_var/Figure2 (another copy)/correct_explicit_final_1/100 N = 64, T = 0.0001, eq_time = 100.0, a = 1.0, b = 1.0, k = 0.5, dt = 0.0005/output.txt')
time_a = data[:,2]
q_numerical_a_3 = (data[:,3] + data[:,5])*10

data = loadtxt('heat_tau_analytical_model_A.txt')
time_a_1 = data[:,0]
q_analitical_a = data[:,1]

data = loadtxt('heat_tau_simulation_model_B.txt')
time_b = data[:,0]
q_numerical_b = data[:,1]

data = loadtxt('heat_tau_analytical_model_B.txt')
time_b_1 = data[:,0]
q_analitical_b = data[:,1]

data = loadtxt('/home/atul/PhD_thesis/Paper_write_OMTP_mean_var/Figure2 (another copy)/correct_explicit_final_1/100 N = 64, T = 0.01, eq_time = 100.0, a = 1.0, b = 1.0, k = 0.5, dt = 0.0005/parameters_only.txt')
wasser = data[15]
delta_F = data[14]
time_a_1 = np.linspace(0.01, 3, 300)
q_analitical_a = wasser/time_a_1

with open('all_cost_function_data_f', 'rb') as fp:
    data = pickle.load(fp)
al_f = data[0]
cost_fn_f = data[1][-3]

with open('der_all_cost_function_data_f', 'rb') as fp:
    data = pickle.load(fp)
al_der_f = data[0][-3]
hyst_der_f = data[1][-3]

with open('all_cost_function_data_r', 'rb') as fp:
    data = pickle.load(fp)
al_r = data[0]
cost_fn_r = data[1][-3]

cost_fn_r_1 = data[1][0]
cost_fn_r_2 = data[1][10]
cost_fn_r_3 = data[1][18]

with open('all_cost_function_data_small_mu', 'rb') as fp:
    data = pickle.load(fp)
al_r_s_mu = data[0]
cost_fn_r_s_mu = data[1][6]
cost_fn_r_1_s_mu = data[1][11]
cost_fn_r_2_s_mu = data[1][16]
cost_fn_r_3_s_mu = data[1][22]



with open('der_all_cost_function_data_r', 'rb') as fp:
    data = pickle.load(fp)
al_der_r = data[0][-3]
hyst_der_r = data[1][-3]

with open('output_data', 'rb') as fp:
    data = pickle.load(fp)
mean = data[0]
std = data[1]

with open('phase_space_all', 'rb') as fp:
    data = pickle.load(fp)
phase1 = data[0]
phase2 = data[3]
phase3 = data[4]
phase4 = data[5]

with open('multitrap_cost_function_hysterisis', 'rb') as fp:
    data = pickle.load(fp)
mult_al_f =data[0][0]
mult_cfn_f =data[0][1]
mult_al_b =data[1][0]
mult_cfn_b =data[1][1]

with open('all_trap_pareto_moment', 'rb') as fp:
    data = pickle.load(fp)
mean_0 = data[0][0]
std_0 = data[0][1]
mean_1 = data[1][0]
std_1 = data[1][1]
mean_2 = data[7][0]
std_2 = data[7][1]
mean_3 = data[12][0]
std_3 = data[12][1]

with open('cost_fn_pareto_moment', 'rb') as fp:
    data = pickle.load(fp)
mean_mult = data[0]
std_mult = data[1]


lw  = 5e-1
lw1 = 5e-1
lw2 = 5e-1
lw3 = 5e-1
ss  = 7
ssl = 5
sss = 5
lt  = 2
p   = 1
w   = 5e-1
tp  = r'$\dfrac{\cal P}{{\cal P}_0}$'

col1 = 'cornflowerblue'
col2 = 'orange'
col3 = 'red'

# ---------
#  Plot 01
# ---------

ax = axes([l01, b01, w01, h01])
ax.tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
ax.tick_params(axis='x', direction='out', length=lt, pad=p, width=w)
#ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useOffset=-100)


for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

lines = []
ax.plot(time_a_1, q_analitical_a/10, color='blue', lw=lw2)
lines += ax.plot(time_a, q_numerical_a_3, '1', markersize=3, color='green', lw=lw1, label=r'$T = 10^{-4}$')
lines += ax.plot(time_a, q_numerical_a_2, '^', markersize=1, color=col1, lw=lw1, label=r'$T = 10^{-3}$')
lines += ax.plot(time_a, q_numerical_a, '.', markersize=2, color=col3, lw=lw1, label=r'$T = 10^{-2}$')
lines += ax.plot(time_a, q_numerical_a_1, 's', markersize=1, color=col2, lw=lw1, label=r'$T = 10^{-1}$')
ax.labels = [l.get_label() for l in lines]
ax.legend(lines, ax.labels, bbox_to_anchor=(0.5, 0.45), fontsize=4.5, frameon=False)


xx  = min(time_a) - 0.25*(max(time_a)-min(time_a))
xxx = max(time_a) - 0.05*(max(time_a)-min(time_a))
yy  = min(q_numerical_a) - 0.25*(max(q_numerical_a)-min(q_numerical_a))
yyy = max(q_numerical_a) + 0.25*(max(q_numerical_a)-min(q_numerical_a))

ax.text(xxx - 0.08*(xxx - xx), yyy - 0.08*(yyy - yy), r'(a)', size=ssl)
ax.text(xxx - 0.1*(xxx- xx), yy + 0.05*(yyy - yy), r'$\tau$', size=ss)
#ax.text(xx + 0.08, 0.9*yyy, r'$\frac{\langle W \rangle - \Delta \cal{F}}{T}$', size=ss)
ax.text(xx + 0.05*(xxx - xx), yyy - 0.2*(yyy - yy), r'$\Sigma$', size=ss)
ax.text(xx-0.35, 0.98*yyy, r'$1e3$', size=ssl)

ax.axis([xx, xxx, yy, yyy])
# ax.set_xticklabels([])
# ax.set_xticks([])
# yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])


# ---------
#  Plot 02
# ---------

i_xx = 0.3
i_xxx = 0.48
i_yy = - 0.1
i_yyy = 0.25

xx = 0.0
xxx = 1.1
yy = - 1.25
yyy = 1.42

ax = axes([l02, b02, w02, h02])
ax.tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
ax.tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)


ratio = 0.49

ax2 = axes([l02 + (1.0-ratio)*w02, b02 + (1.0-ratio)*h02, ratio*w02, ratio*h02])


ax2.tick_params(axis='y', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)
ax2.tick_params(axis='x', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)

for i, line in enumerate(ax2.get_xticklines() + ax2.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

# ax2.set_yticklabels([])
# ax2.set_xticklabels([])

ax2.plot(mean, std, '-', '.', markersize=3, color=col3, lw=lw1)

# xx  = min(al_f) - 0.15*(max(al_f)-min(al_f))
# xxx = max(time_b) + 0.05*(max(time_b)-min(time_b))
# yy  = min(cost_fn_f) - 0.15*(max(cost_fn_f)-min(cost_fn_f))
# yyy = max(cost_fn_r) + 0.15*(max(cost_fn_r)-min(cost_fn_r))

# i_xx = 0.3
# i_xxx = 0.8
# i_yy = - 1.5
# i_yyy = 0.25

ax2.text(0.93*i_xxx, 0.72*i_yyy, r'(b)', size=ssl)
# ax2.text(0.9*i_xxx, i_yy + 0.02,  r'$\langle W \rangle$', size=1.5*ratio*ss)
# ax2.text(0.75*i_xxx, 0.4*i_yyy, r'$\langle W \rangle$', size=1.5*ratio*ss)
# ax2.text(i_xx + 0.005, 0.7*i_yyy-0.04, r'$\sigma$', size=1.5*ratio*ss)
# ax2.plot([0.43, 0.43], [0, -2], ':', color='black', alpha=.8, lw=2*lw)


ax2.axis([i_xx, i_xxx, i_yy, i_yyy])

# lines = []
# lines += ax2.plot(al_r_s_mu, cost_fn_r_s_mu, '-', markersize=2.5, marker='1', color=col1, lw=lw1, label=r'$\mu$ = 0.05')
# lines += ax2.plot(al_r_s_mu, cost_fn_r_1_s_mu, '-', marker='s', markersize=0.75, color=col2, lw=lw1, label=r'$\mu$ = 0.1')
# lines += ax2.plot(al_r_s_mu, cost_fn_r_2_s_mu, '-', marker='^', markersize=0.75, color=col3, lw=lw1, label=r'$\mu$ = 0.2')
# lines += ax2.plot(al_r_s_mu, cost_fn_r_3_s_mu, '-', marker='.', markersize=1, color='green', lw=lw1, label=r'$\mu$ = 0.4')
# ax2.labels = [l.get_label() for l in lines]
# ax2.legend(lines, ax2.labels, bbox_to_anchor=(0.56, 0.64), fontsize=3.5, frameon=False)


lines = []
lines += ax2.plot(al_r, cost_fn_r_1, '-', markersize=0.75, marker='s', color=col2, lw=1.2*lw1, label=r'$\mu$ = 0.05')
lines += ax2.plot(al_r, cost_fn_r_2, '-', marker='^', markersize=0.75, color=col3, lw=1.2*lw1, label=r'$\mu$ = 0.1')
lines += ax2.plot(al_r, cost_fn_r_3, '-', marker='.', markersize=1, color='green', lw=1.2*lw1, label=r'$\mu$ = 0.2')
lines += ax2.plot(al_r, cost_fn_r, '-', marker='1', markersize=2.5, color=col1, lw=1.2*lw1, label=r'$\mu$ = 0.4')
# ax2.labels = [l.get_label() for l in lines]
# ax2.legend(lines, ax2.labels, bbox_to_anchor=(0.56, 0.64), fontsize=3.5, frameon=False)


lines = []
lines += ax.plot(al_r, cost_fn_r_1, '-', marker='s', markersize=1, color=col2, lw=1.2*lw1, label=r'$\mu$ = 1.0')
lines += ax.plot(al_r, cost_fn_r_2, '-', marker='^', markersize=1, color=col3, lw=1.2*lw1, label=r'$\mu$ = 2.1')
lines += ax.plot(al_r, cost_fn_r_3, '-', marker='.', markersize=2.5, color='green', lw=1.2*lw1, label=r'$\mu$ = 2.8')
lines += ax.plot(al_r, cost_fn_r, '-', markersize=3, marker='1', color=col1, lw=1.2*lw1, label=r'$\mu$ = 4.0')
ax.labels = [l.get_label() for l in lines]
ax.legend(lines, ax.labels, bbox_to_anchor=(0.4, 0.45), fontsize=4.5, frameon=False)

ax.plot([i_xx, (1-ratio)*(xxx-xx)], [i_yyy, yyy], '-', color='grey', alpha=.8, lw=1.5*lw)
ax.plot([i_xxx, xxx], [i_yy, (1-ratio)*(yyy-yy) + yy], '-', color='grey', alpha=.8, lw=1.5*lw)

# plot(al_r, cost_fn_r, '.-', markersize=1,color=col2, lw=lw2)


# xx  = (min(al_r) - 0.05*(max(al_r)-min(al_r)))
# xxx = (max(al_r) + 0.05*(max(al_r)-min(al_r)))
# yy  = min(cost_fn_r) - 0.2*(max(cost_fn_r)-min(cost_fn_r))
# yyy = 1.2*(max(cost_fn_r_1) + 0.10*(max(cost_fn_r_1)-min(cost_fn_r_1)))

#ax.text(0.9*xxx, 0.9*yyy, r'(b)', size=ssl)
ax.text(0.78*xxx, 0.9*yy, r'$\lambda$', size=ss)
ax.text(1.1*xx+0.03, 0.7*yyy, r'$\cal J$$^\lambda_0$', size=ss)
ax.annotate('', xy=(-3.8, 3.5), xytext=(-1.9, 3.5), arrowprops=dict(arrowstyle='<->', color='black'))
ax.axvline(x=-3.8, ymin=3.0, ymax=3.3)
ax.axvline(x=-1.7, ymin=3.0, ymax=3.3)

ax.plot([i_xx, i_xxx], [i_yy, i_yy], '--', color='gray', alpha=.8, lw=1.5*lw)
ax.plot([i_xx, i_xxx], [i_yyy, i_yyy], '--', color='gray', alpha=.8, lw=1.5*lw)
ax.plot([i_xx, i_xx], [i_yy, i_yyy], '--', color='gray', alpha=.8, lw=1.5*lw)
ax.plot([i_xxx, i_xxx], [i_yy, i_yyy], '--', color='gray', alpha=.8, lw=1.5*lw)

ax.axis([xx, xxx, yy, yyy])
# xticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])
# yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])


# ---------
#  Plot 11
# ---------

i_xx  = 0.33
i_xxx = 0.53
i_yy  = -0.19
i_yyy = 0.22

xx  = (min(al_der_f) - 0.05*(max(al_der_f)-min(al_der_f)))
xxx = 1.2*(max(al_der_f) + 0.05*(max(al_der_f)-min(al_der_f))) + 0.2
yy  = min(hyst_der_f) - 0.2*(max(hyst_der_f)-min(hyst_der_f))
yyy = 1.2*(max(hyst_der_f) + 0.10*(max(hyst_der_f)-min(hyst_der_f))) + 0.6

ax = axes([l11, b11, w11, h11])
ax.tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
ax.tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)


ratio = 0.5

ax2 = axes([l11 + (1-ratio)*w11, b11 + (1-ratio)*h11 - 0.02, ratio*w11, ratio*h11])


ax2.tick_params(axis='y', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)
ax2.tick_params(axis='x', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)

# ax.text(xx + 0.02, 1.3*yyy - 0.7, r'$d$$\cal J$$^0_\lambda / d \lambda$', size=ss)
ax.text(xx + 0.02, 1.3*yyy - 0.7, r'$d$$\cal J$$^\lambda_0 / d \lambda$', size=ss)
# ax.text(xx + 0.02, 1.3*yyy - 0.7, r'$d$$\cal J$$_0 / d \lambda$', size=ss)
ax.plot([0.53, 0.53], [-0.2, -1.9], '--', color='gray', alpha=.8, lw=1.5*lw)
ax.plot([0.34, 0.34], [-0.2, -1.9], '--', color='gray', alpha=.8, lw=1.5*lw)
ax.plot([0.53, 0.34], [-0.2, -0.2], '--', color='gray', alpha=.8, lw=1.5*lw)
ax.plot([0.53, 0.34], [-1.9, -1.9], '--', color='gray', alpha=.8, lw=1.5*lw)
ax.plot([(1-ratio)*(xxx-xx)+xx, 0.34], [yyy-0.15, -0.2], '-', color='grey', alpha=.8, lw=1.5*lw)
ax.plot([xxx, 0.53], [(1-ratio)*(yyy-yy)+yy - 0.1, -1.9], '-', color='grey', alpha=.8, lw=1.5*lw)

for i, line in enumerate(ax2.get_xticklines() + ax2.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

# ax2.set_yticklabels([])
# ax2.set_xticklabels([])

ax2.plot(al_f, cost_fn_f, '.-', markersize=1.5, color=col1, lw=lw1)
ax2.plot(al_r, cost_fn_r, '.-', markersize=1.5, color=col2, lw=lw2)

# xx  = min(al_f) - 0.15*(max(al_f)-min(al_f))
# xxx = max(time_b) + 0.05*(max(time_b)-min(time_b))
# yy  = min(cost_fn_f) - 0.15*(max(cost_fn_f)-min(cost_fn_f))
# yyy = max(cost_fn_r) + 0.15*(max(cost_fn_r)-min(cost_fn_r))

#ax2.text(0.50, 0.7*i_yyy, r'(c)', size=ssl)
ax2.text(0.9*i_xxx, i_yy + 0.02, r'$\lambda$', size=1.5*ratio*ss)
ax2.text(0.8*i_xxx, -0.265, r'$\lambda_c$', size=1.5*ratio*ss)
# ax2.text(i_xx + 0.005, 0.7*i_yyy-0.04, r'$\cal J$$_0$'.format(str(2)), size=1.5*ratio*ss)
ax2.text(i_xx + 0.005, 0.7*i_yyy-0.04, r'$\cal J$$^\lambda_0$'.format(str(2)), size=1.5*ratio*ss)
ax2.plot([0.43, 0.43], [0, -2], ':', color='black', alpha=.8, lw=2*lw)


ax2.axis([i_xx, i_xxx, i_yy, i_yyy])


ax.plot(al_der_f, hyst_der_f, '.-', markersize=1.5, color=col1, lw=lw1)
ax.plot(al_der_r, hyst_der_r, '.-', markersize=1.5, color=col2, lw=lw2)

ax.text(0.9*xxx, 0.7*yyy - 0.2, r'(d)', size=ssl)
ax.text(0.9*xxx, yy + 0.08, r'$\lambda$', size=ss)


ax.axis([xx, xxx, yy, yyy])
#xticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])
#yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])


# ---------
#  Plot 12
# ---------


ax = axes([l12, b12, w12, h12])
ax.tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
ax.tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)


ratio = 0.5

#ax2 = axes([l12 + (0.98-ratio)*w12, b12 + (0.97-ratio)*h12, ratio*w12, ratio*h12])


# ax2.tick_params(axis='y', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)
# ax2.tick_params(axis='x', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)

# for i, line in enumerate(ax2.get_xticklines() + ax2.get_yticklines()):
#	if i%2==1:
#		line.set_visible(False)

# ax2.set_yticklabels([])
# ax2.set_xticklabels([])

# ax2.plot(al_f, cost_fn_f, '.-', markersize=1, color=col1, lw=lw1)
# ax2.plot(al_r, cost_fn_r, '.-', markersize=1, color=col2, lw=lw2)

# xx  = min(al_f) - 0.15*(max(al_f)-min(al_f))
# xxx = max(time_b) + 0.05*(max(time_b)-min(time_b))
# yy  = min(cost_fn_f) - 0.15*(max(cost_fn_f)-min(cost_fn_f))
# yyy = max(cost_fn_r) + 0.15*(max(cost_fn_r)-min(cost_fn_r))

# i_xx  = -0.8
# i_xxx = 1.0
# i_yy  = 0.0
# i_yyy = 8

# ax2.text(i_xxx - 0.15*(i_xxx - i_xx), i_yyy - 0.15*(i_yyy - i_yy), r'(d)', size=ssl)
# ax2.text(0.7*i_xxx, i_yy + 0.2, r'$\lambda_c$', size=1.5*ratio*ss)
# ax2.text(0.75*i_xxx, 0.4*i_yyy, r'$\lambda = \lambda_c$', size=1.5*ratio*ss)
# ax2.text(i_xx + 0.05, 0.85*i_yyy-0.04, r'$\mu$'.format(str(2)), size=1.5*ratio*ss)
# ax2.plot([0.43, 0.43], [0, -2], ':', color='black', alpha=.8, lw=2*lw)

# ax2.axis([i_xx, i_xxx, i_yy, i_yyy])

lines = []
lines += ax.plot(phase1[0][:-1], phase1[1][:-1], '-', marker='1', markersize=2, color=col1, lw=lw1, label=r'q = 0')
#lines += ax2.plot(phase2[0], phase2[1], '-', marker='s', markersize=0.75, color=col2, lw=lw1, label=r'q = 0.39')
#lines += ax2.plot(phase3[0][:-1], phase3[1][:-1], '-', marker='^', markersize=0.75, color=col3, lw=lw1, label=r'q = 0.79')
#lines += ax2.plot(phase4[0][:-1], phase4[1][:-1], '-', marker='.', markersize=1.5, color='green', lw=lw1, label=r'q = 1.57')
ax.labels = [l.get_label() for l in lines]
#ax.legend(lines, ax.labels, bbox_to_anchor=(0.65, 0.8), fontsize=4, frameon=False)

xx  = 0
xxx = 1.0
yy  = 0.0
yyy = 8

ax.text(0.8*xxx, yy + 0.2, r'$\lambda$', size=ss)
ax.text(xx + 0.05, 0.85*yyy, r'$\mu$', size=ss)
ax.text(xxx - 0.08*(xxx - xx), yyy - 0.08*(yyy - yy), r'(d)', size=ssl)

#X1 = linspace(0, phase1[0][0][0], 100)
#X2 = linspace(phase1[0][0][0], 1, 100)
#X3 = linspace(0, 1, 100)
#Y1 = phase1[1][0] + 0*X1
#Y2 = yyy + 0*X1
#Y3 = phase1[1][0] + 0*X2
#Y4 = yyy + 0*X2
#Y5 = phase1[1][0] + 0*X3
#Y6 = 0 + 0*X3


#ax.plot(X3, Y5, color='black', lw=lw1)
#ax.plot(X1, Y2, color='black', lw=lw1)
#ax.plot(X2, Y3, color='black', lw=lw1)
# ax.plot(X2, Y3, Y4, color='black', lw=lw1)
#ax.plot([phase1[0][0], phase1[0][0]], [phase1[1][0], yyy], '-', color='black', lw=lw1)

#ax.fill_between(X1, Y1, Y2, color='red', alpha=.3)
#ax.fill_between(X2, Y3, Y4, color='blue', alpha=.3)
#ax.fill_between(X3, Y5, Y6, color='yellow', alpha=.3)

ax.text(0.06, 4, ' dominated', size=1.2*ssl)
ax.text(0.17, 4.8, r'$\sigma_q$', size=ssl)
ax.text(0.53, 4, 'dominated', size=1.2*ssl)
ax.text(0.61, 4.9, r'$\langle W_q \rangle$', size=ssl)
#ax.text(0.2, 1.0, 'No phase transition', size=1.2*ssl)


ax.axis([xx, xxx, yy, yyy])
# xticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])
# yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])



# ---------
#  Plot 21
# ---------

ax = axes([l21, b21, w21, h21])
tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

k = 18
plot(mean[:k], std[:k], '.-', markersize=2.5, color=col3, lw=2*lw1)
plot(mean[k:], std[k:], '.-', markersize=2.5, color=col3, lw=2*lw1)

# plot(al_r, cost_fn_r, '.-', markersize=1,color=col2, lw=lw2)

plot([-3.7, -3.7], [3.5, 2.9], ':', color='black', alpha=.8, lw=2*lw)
plot([-2.0, -2.0],[3.5, 1.5], ':', color='black', alpha=.8, lw=2*lw)

xx  = min(mean) - 0.15*(max(mean)-min(mean))
xxx = max(mean) + 0.05*(max(mean)-min(mean))
yy  = min(std) - 0.15*(max(std)-min(std))
yyy = max(std) + 0.15*(max(std)-min(std))

#text(0.9*xxx-0.4, 0.9*yyy, r'(e)', size=ssl)
text(-3.4, 3.0, r'$\lambda = \lambda_c$', size=ss)
text(0.7*xxx - 2.0, yy + 0.2, r'$\langle W_0 \rangle$', size=ss)
text(xx + 0.1, 0.7*yyy, r'$\sigma_0$', size=ss)
annotate('', xy=(-3.8, 3.5), xytext=(-1.9, 3.5), arrowprops=dict(arrowstyle='<->', color='black'))
axvline(x=-3.8, ymin=3.0, ymax=3.3)
axvline(x=-1.7, ymin=3.0, ymax=3.3)


axis([xx, xxx, yy, yyy])
# xticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])
# yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])


# ---------
#  Plot 22
# ---------

#ax = axes([l22, b22, w22, h22])
#ax.tick_params(axis='y', direction='in', length=lt, pad=p, width=w)
#ax.tick_params(axis='x', direction='in', length=lt, pad=p, width=w)

xx  = 0
xxx = 1
yy  = 0
yyy = 1

#ax.set_yticklabels([])
#ax.set_xticklabels([])

#ax.axis([xx, xxx, yy, yyy])

# for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
# 	if i%2==1:
#		line.set_visible(False)


ratio = 0.46

#ax = axes([l22, b22, w22, h22])

#for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
#	if i%2==1:
#		line.set_visible(False)

ax4 = axes([l22, b22 + (1-ratio)*h22, ratio*w22, ratio*h22])


ax4.tick_params(axis='y', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)
ax4.tick_params(axis='x', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)

for i, line in enumerate(ax4.get_xticklines() + ax4.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

l = 10
ax4.plot(mean_0[:l], std_0[:l], '-', markersize=2, color=col1, marker='1', lw=lw1)
ax4.plot(mean_0[l:], std_0[l:], '-', markersize=2, color=col1, marker='1', lw=lw1)

# xx  = min(al_f) - 0.15*(max(al_f)-min(al_f))
# xxx = max(time_b) + 0.05*(max(time_b)-min(time_b))
# yy  = min(cost_fn_f) - 0.15*(max(cost_fn_f)-min(cost_fn_f))
# yyy = max(cost_fn_r) + 0.15*(max(cost_fn_r)-min(cost_fn_r))

i_xx  = min(mean_0) - 0.4*(max(mean_0)-min(mean_0))
i_xxx = max(mean_0) + 0.25*(max(mean_0)-min(mean_0))
i_yy  = min(std_0) - 0.45*(max(std_0)-min(std_0))
i_yyy = max(std_0) + 0.15*(max(std_0)-min(std_0))

ax4.text(i_xxx - 0.3*(i_xxx - i_xx), i_yy + 0.08*(i_yyy - i_yy),  r'$\langle W_0 \rangle$', size=1.2*ratio*ss)
ax4.text(i_xx + 0.05*(i_xxx - i_xx), i_yyy - 0.3*(i_yyy - i_yy), r'$\sigma_0^2$', size=1.5*ratio*ss)
ax4.text(i_xxx - 0.15*(i_xxx - i_xx), i_yyy - 0.15*(i_yyy - i_yy), r'(1)', size=ssl)

ax4.axis([i_xx, i_xxx, i_yy, i_yyy])


ax2 = axes([l22 + (1-ratio)*w22, b22 + (1-ratio)*h22, ratio*w22, ratio*h22])


ax2.tick_params(axis='y', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)
ax2.tick_params(axis='x', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)

for i, line in enumerate(ax2.get_xticklines() + ax2.get_yticklines()):
	if i%2==1:
		line.set_visible(False)


l = 9
ax2.plot(mean_1[:l], std_1[:l], '-', markersize=0.75, marker='s', color=col2, lw=lw1)
ax2.plot(mean_1[l:], std_1[l:], '-', markersize=0.75, marker='s', color=col2, lw=lw1)

# xx  = min(al_f) - 0.15*(max(al_f)-min(al_f))
# xxx = max(time_b) + 0.05*(max(time_b)-min(time_b))
# yy  = min(cost_fn_f) - 0.15*(max(cost_fn_f)-min(cost_fn_f))
# yyy = max(cost_fn_r) + 0.15*(max(cost_fn_r)-min(cost_fn_r))

i_xx  = min(mean_1) - 0.4*(max(mean_1)-min(mean_1))
i_xxx = max(mean_1) + 0.25*(max(mean_1)-min(mean_1))
i_yy  = min(std_1) - 0.45*(max(std_1)-min(std_1))
i_yyy = max(std_1) + 0.15*(max(std_1)-min(std_1))

ax2.text(i_xxx - 0.3*(i_xxx - i_xx), i_yy + 0.08*(i_yyy - i_yy),  r'$\langle W_1 \rangle$', size=1.2*ratio*ss)
ax2.text(i_xx + 0.05*(i_xxx - i_xx), i_yyy - 0.3*(i_yyy - i_yy), r'$\sigma_1^2$', size=1.5*ratio*ss)
ax2.text(i_xxx - 0.15*(i_xxx - i_xx), i_yyy - 0.15*(i_yyy - i_yy), r'(2)', size=ssl)

ax2.axis([i_xx, i_xxx, i_yy, i_yyy])


ax5 = axes([l22, b22, ratio*w22, ratio*h22])


ax5.tick_params(axis='y', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)
ax5.tick_params(axis='x', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)

for i, line in enumerate(ax5.get_xticklines() + ax5.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

ax5.plot(mean_2, std_2, '-', markersize=1, marker='^', color='green', lw=lw1)

# xx  = min(al_f) - 0.15*(max(al_f)-min(al_f))
# xxx = max(time_b) + 0.05*(max(time_b)-min(time_b))
# yy  = min(cost_fn_f) - 0.15*(max(cost_fn_f)-min(cost_fn_f))
# yyy = max(cost_fn_r) + 0.15*(max(cost_fn_r)-min(cost_fn_r))

i_xx  = min(mean_2) - 0.4*(max(mean_2)-min(mean_2))
i_xxx = max(mean_2) + 0.25*(max(mean_2)-min(mean_2))
i_yy  = min(std_2) - 0.45*(max(std_2)-min(std_2))
i_yyy = max(std_2) + 0.15*(max(std_2)-min(std_2))

ax5.text(i_xxx - 0.3*(i_xxx - i_xx), i_yy + 0.08*(i_yyy - i_yy),  r'$\langle W_7 \rangle$', size=1.2*ratio*ss)
ax5.text(i_xx + 0.05*(i_xxx - i_xx), i_yyy - 0.3*(i_yyy - i_yy), r'$\sigma_7^2$', size=1.5*ratio*ss)
ax5.text(i_xxx - 0.15*(i_xxx - i_xx), i_yyy - 0.15*(i_yyy - i_yy), r'(3)', size=ssl)


ax5.axis([i_xx, i_xxx, i_yy, i_yyy])


ax3 = axes([l22 + (1-ratio)*w22, b22, ratio*w22, ratio*h22])


ax3.tick_params(axis='y', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)
ax3.tick_params(axis='x', direction='in', length=ratio*lt, pad=ratio*p, width=w, labelsize=5)

for i, line in enumerate(ax3.get_xticklines() + ax3.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

# ax3.set_yticklabels([])
# ax3.set_xticklabels([])

l = 9
ax3.plot(mean_mult[:l], std_mult[:l], '-', markersize=2, marker='.', color=col3, lw=lw1)
ax3.plot(mean_mult[l+1:], std_mult[l+1:], '-', markersize=2, marker='.', color=col3, lw=lw1)

# xx  = min(al_f) - 0.15*(max(al_f)-min(al_f))
# xxx = max(time_b) + 0.05*(max(time_b)-min(time_b))
# yy  = min(cost_fn_f) - 0.15*(max(cost_fn_f)-min(cost_fn_f))
# yyy = max(cost_fn_r) + 0.15*(max(cost_fn_r)-min(cost_fn_r))

i_xx  = min(mean_mult) - 0.4*(max(mean_mult)-min(mean_mult))
i_xxx = max(mean_mult) + 0.25*(max(mean_mult)-min(mean_mult))
i_yy  = min(std_mult) - 0.45*(max(std_mult)-min(std_mult))
i_yyy = max(std_mult) + 0.15*(max(std_mult)-min(std_mult))

ax3.text(i_xxx - 0.3*(i_xxx - i_xx), i_yy + 0.08*(i_yyy - i_yy),  r'$\langle W \rangle$', size=1.2*ratio*ss)
ax3.text(i_xx + 0.05*(i_xxx - i_xx), i_yyy - 0.2*(i_yyy - i_yy), r'$\sigma^2$', size=1.5*ratio*ss)
ax3.text(i_xxx - 0.15*(i_xxx - i_xx), i_yyy - 0.15*(i_yyy - i_yy), r'(4)', size=ssl)


ax3.axis([i_xx, i_xxx, i_yy, i_yyy])


# lines = []
# lines += ax.plot(mean_3, std_3, '.', marker='.', markersize=3, color='green', lw=lw1, label=r'q = 2.36')
# lines += ax.plot(mean_2, std_2, '.', marker='^', markersize=1.5, color=col3, lw=lw1, label=r'q = 1.57')
# lines += ax.plot(mean_1, std_1, '.', marker='s', markersize=1.5, color=col2, lw=lw1, label=r'q = 0.59')
# lines += ax.plot(mean_0, std_0, '.', markersize=3, marker='1', color=col1, lw=lw1, label=r'q = 0.20')
# ax.labels = [l.get_label() for l in lines]
# ax.legend(lines, ax.labels, bbox_to_anchor=(0.52, 0.49), fontsize=5, frameon=False)

# plot(al_r, cost_fn_r, '.-', markersize=1,color=col2, lw=lw2)

# ax.plot([-3.7, -3.7], [3.5, 2.9], ':', color='black', alpha=.8, lw=2*lw)
# ax.plot([-2.0, -2.0],[3.5, 1.5], ':', color='black', alpha=.8, lw=2*lw)

xx  = min(mean_0) - 0.15*(max(mean_0)-min(mean_0))
xxx = max(mean_0) + 0.05*(max(mean_0)-min(mean_0)) + 0.2
yy  = min(std_0) - 0.15*(max(std_0)-min(std_0)) - 1.5
yyy = max(std_0) + 0.15*(max(std_0)-min(std_0)) + 1.0

#ax.axes([l22, b22, w22, h22])

# ax.text(0.9*xxx-0.1, 0.9*yyy, r'(f)', size=ssl)
# ax.text(-3.4, 3.0, r'$\lambda = \lambda_c$', size=ss)
# ax.text(xxx - 0.4, yy + 0.3, r'$\langle W_q \rangle$', size=ss)
# ax.text(xx+0.1, 0.85*yyy, r'$\sigma_q$', size=ss)
# ax.annotate('', xy=(-3.8, 3.5), xytext=(-1.9, 3.5), arrowprops=dict(arrowstyle='<->', color='black'))
# ax.axvline(x=-3.8, ymin=3.0, ymax=3.3)
# ax.axvline(x=-1.7, ymin=3.0, ymax=3.3)


# ax.axis([xx, xxx, yy, yyy])
# xticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])
# yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])


# ---------
#  Plot 31
# ---------

ax = axes([l31, b31, w31, h31])
tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

plot(mult_al_f, mult_cfn_f, '.-', markersize=1, color=col1, lw=lw1)
plot(mult_al_b, mult_cfn_b, '.-', markersize=1, color=col2, lw=lw2)

xx  = min(mult_al_f) - 0.15*(max(mult_al_f)-min(mult_al_f))
xxx = max(mult_al_f) + 0.05*(max(mult_al_f)-min(mult_al_f))
yy  = min(mult_cfn_f) - 0.15*(max(mult_cfn_f)-min(mult_cfn_f))
yyy = max(mult_cfn_f) + 0.15*(max(cost_fn_r)-min(mult_cfn_f))

text(0.9*xxx, 0.9*yyy - 1, r'(g)', size=ssl)
text(0.9*xxx - 0.1, 0.9*yy, r'$\lambda$', size=ss)
text(xx + 0.05, 0.7*yyy - 4, r'$\cal J_\lambda$', size=ss)

axis([xx, xxx, yy, yyy])
# xticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])
# yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])


# ---------
#  Plot 32
# ---------


ax = axes([l32, b32, w32, h32])
tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

plot(al_der_f, hyst_der_f, '.-', markersize=1, color=col1, lw=lw1)
plot(al_der_r, hyst_der_r, '.-', markersize=1, color=col2, lw=lw2)

xx  = min(al_der_f) - 0.15*(max(al_der_f)-min(al_der_f))
xxx = max(al_der_f) + 0.05*(max(al_der_f)-min(al_der_f))
yy  = min(hyst_der_f) - 0.15*(max(hyst_der_f)-min(hyst_der_f))
yyy = max(hyst_der_f) + 0.15*(max(hyst_der_f)-min(hyst_der_f))

text(0.9*xxx, 0.9*yyy, r'(b)', size=ssl)
text(0.9*xxx, yy, r'$\tau$', size=ss)
text(xx, 0.85*yyy, r'$\cal Q$', size=ss)

axis([xx, xxx, yy, yyy])
# xticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])
# yticks([-1, 0, 1], [r'$-1$', r'$0$', r'$1$'])




# Save figure
filename = 'fig_poster_a.pdf'
savefig(filename, dpi=3e2)

