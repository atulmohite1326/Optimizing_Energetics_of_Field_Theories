import numpy as np
from pylab import *
from matplotlib.ticker import  *
from matplotlib.pyplot import  *
from numpy import *
from matplotlib import cm
#import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from PIL import Image
import pickle
#from pdf2image import convert_from_path


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

space         = 0.07
nb_lines      = 1
fig_width_pt  = 246.
inches_per_pt = 1./72.
golden_mean   = .66
fig_width     = fig_width_pt*inches_per_pt
fig_height    = (fig_width*golden_mean)+space
fig_size      = [fig_width, fig_height]
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

l01 = 7e-2
b01 = 0.58
w01 = .42
h01 = .4

l02 = l01
b02 = 10e-2
w02 = w01
h02 = h01

l11 = .55
b11 = b01
w11 = w01
h11 = h01

l12 = l11
b12 = b02
w12 = w11
h12 = h11

l21 = 1.70
b21 = b11
w21 = w11
h21 = h11

l22 = l21
b22 = b12
w22 = w11
h22 = h11

l31 = 1.9
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

data = loadtxt('heat_tau_simulation_model_A.txt')
time_a = data[:,0]
q_numerical_a = data[:,1]

data = loadtxt('heat_tau_simulation_model_A.txt')
time_a = data[:,0]
q_numerical_a_1 = data[:,1]

data = loadtxt('heat_tau_simulation_model_A.txt')
time_a = data[:,0]
q_numerical_a_2 = data[:,1]

data = loadtxt('heat_tau_simulation_model_A.txt')
time_a = data[:,0]
q_numerical_a_3 = data[:,1]

data = loadtxt('heat_tau_analytical_model_A.txt')
time_a_1 = data[:,0]
q_analitical_a = data[:,1]

data = loadtxt('heat_tau_simulation_model_B.txt')
time_b = data[:,0]
q_numerical_b = data[:,1]

data = loadtxt('heat_tau_analytical_model_B.txt')
time_b_1 = data[:,0]
q_analitical_b = data[:,1]

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
protocols = data[2]
# alpha = data[4]
alpha = al_f

with open('phase_space_all', 'rb') as fp:
    data = pickle.load(fp)
phase1 = data[0]
phase2 = data[3]
phase3 = data[4]
phase4 = data[5]

with open('phase_space_all_model_B_converged_one', 'rb') as fp:
    data = pickle.load(fp)
phase1_b = data[0]
phase2_b = data[1]
phase3_b = data[2]
phase4_b = data[3]


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
lt  = 4
p   = 1
w   = 5e-1
tp  = r'$\dfrac{\cal P}{{\cal P}_0}$'

col1 = 'cornflowerblue'
col2 = 'orange'
col3 = 'red'

col_array = [col1, col2, col3]



# ---------
#  Plot 01
# ---------

i_xx = 0.3
i_xxx = 0.48
i_yy = - 0.1
i_yyy = 0.25

xx = -0.1
xxx = 1.1
yy = - 10.25
yyy = 1.42

ax = axes([l01, b01, w01, h01])
ax.tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
ax.tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)


ratio = 0.49


i_xx = 0.3
i_xxx = 0.8
i_yy = - 1.5
i_yyy = 0.25


lines = []
lines += ax.plot(al_r_s_mu, cost_fn_r_s_mu, '-', markersize=2.5, marker='1', color=col1, lw=lw1, label=r'$\mu$ = 0.05')
lines += ax.plot(al_r_s_mu, cost_fn_r_1_s_mu, '-', marker='s', markersize=0.75, color=col2, lw=lw1, label=r'$\mu$ = 0.1')
lines += ax.plot(al_r_s_mu, cost_fn_r_2_s_mu, '-', marker='^', markersize=0.75, color=col3, lw=lw1, label=r'$\mu$ = 0.2')
lines += ax.plot(al_r_s_mu, cost_fn_r_3_s_mu, '-', marker='.', markersize=1, color='green', lw=lw1, label=r'$\mu$ = 0.4')
ax.labels = [l.get_label() for l in lines]
ax.legend(lines, ax.labels, bbox_to_anchor=(0.6, 0.6), fontsize=4.5, frameon=False)


ax.text(0.72*xxx, 0.9*yy, r'$\lambda$', size=ss)
ax.text(1.1*xx+0.03, -2, r'$\cal J$$^0_\lambda$', size=ss)
ax.text(xxx - 0.1*(xxx-xx), yyy - 0.1*(yyy - yy), r'(a)', size=ssl)


ax.axis([xx, xxx, yy, yyy])


# ---------
#  Plot 02
# ---------

ax2 = axes([l02, b02, w02, h02])
ax2.tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
ax2.tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax2.get_xticklines() + ax2.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

t2 = np.linspace(0, 1, 101)
t3 = np.linspace(-0.1, 0, 2)
t4 = np.linspace(1.0, 1.1, 2)
prt_in = np.asarray([1, 1])
prt_fnl = np.asarray([0.04, 0.04])

lines = []
print(len(al_f))
print(len(protocols))
for i in range(len(protocols)):
    if i == 0:
	    lines += ax2.plot(t2, protocols[i]/20, '-', color=(al_f[i], 0, (1 - al_f[i])), lw=lw1, label=r'$\lambda = 0$')
    elif i == len(protocols) - 1:
	    lines += ax2.plot(t2, protocols[i]/20, '-', color=(al_f[i], 0, (1 - al_f[i])), lw=lw1, label=r'$\lambda = 1$')
    else:
        ax2.plot(t2, protocols[i]/20, '-', color=(al_f[i], 0, (1 - al_f[i])), lw=lw1)

ax2.labels = [l.get_label() for l in lines]
ax2.legend(lines, ax2.labels, bbox_to_anchor=(0.85, 0.9), fontsize=4.5, frameon=False)
#for i in range(len(protocols)):
#	ax.scatter(t2, protocols[i]/20, s=1*np.ones(101), c=al_f[i]*np.ones(101), cmap='viridis')

# n_lines = len(al_f)

# ys = np.array([prtc/20 for prtc in protocols])
# xs = np.array([t2 for i in range(n_lines)]) # could also use np.tile

# colors = np.arange(n_lines)

# lc = multiline(xs, ys, al_f, cmap='rainbow', lw=lw1)

# axcb = fig.colorbar(lc)
# axcb.set_label(r'$\lambda$', labelpad=0,y=0.5)
#ax.set_title('Line Collection with mapped colors')
#ax.plot(t3, prt_in, '--', color='green', lw=lw1)
#ax.plot(t4, prt_fnl, '--', color='green', lw=lw1)

xx = -0.15
xxx = 1.15
yy = -0.15
yyy = 1.25


ax2.text(0.72*xxx, yy + 0.03*(yyy - yy), r'$t/\tau$', size=ss)
ax2.text(xx  + 0.01*(xxx-xx), yyy - 0.2*(yyy - yy), r'$\frac{K^*(0, t)}{K_i(0)}$', size=0.7*ss)
#ax2.text(xxx - 0.1*(xxx-xx), yyy - 0.1*(yyy - yy), r'(b)', size=ssl)

ax2.axis([xx, xxx, yy, yyy])



# ---------
#  Plot 11
# ---------


ax = axes([l11, b11, w11, h11])
ax.tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
ax.tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

xx  = -0.4
xxx = 1.0
yy  = 0.0
yyy = 10.5


lines = []
lines += ax.plot(phase1[0], phase1[1], '-', marker='1', markersize=2, color=col1, lw=lw1, label=r'q = 0')
#lines += ax2.plot(phase2[0], phase2[1], '-', marker='s', markersize=0.75, color=col2, lw=lw1, label=r'q = 0.39')
lines += ax.plot(phase3[0], phase3[1], '-', marker='^', markersize=0.75, color=col3, lw=lw1, label=r'q = 0.79')
lines += ax.plot(phase4[0], phase4[1], '-', marker='.', markersize=1.5, color='green', lw=lw1, label=r'q = 1.57')
ax.labels = [l.get_label() for l in lines]
ax.legend(lines, ax.labels, bbox_to_anchor=(0.45, 0.8), fontsize=4.5, frameon=False)

ax.text(0.8*xxx, yy + 0.2, r'$\lambda$', size=ss)
ax.text(xx + 0.05, 0.85*yyy, r'$\mu$', size=ss)
#ax.text(xxx - 0.1*(xxx-xx), yyy - 0.1*(yyy - yy), r'(c)', size=ssl)

ax.axis([xx, xxx, yy, yyy])


# ---------
#  Plot 12
# ---------


ax = axes([l12, b12, w12, h12])
ax.tick_params(axis='y', direction='out', length=lt, pad=p, width=w)
ax.tick_params(axis='x', direction='out', length=lt, pad=p, width=w)

for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
	if i%2==1:
		line.set_visible(False)

xx  = -0.4
xxx = 1.0
yy  = 0.0
yyy = 10.5


lines = []
lines += ax.plot(phase1_b[0][0:-10], phase1_b[1][0:-10], '-', marker='1', markersize=2, color=col1, lw=lw1, label=r'q = 0.59')
#lines += ax2.plot(phase2[0], phase2[1], '-', marker='s', markersize=0.75, color=col2, lw=lw1, label=r'q = 0.39')
lines += ax.plot(phase2_b[0][0:-10], phase2_b[1][0:-10], '-', marker='^', markersize=0.75, color=col3, lw=lw1, label=r'q = 0.79')
lines += ax.plot(phase3_b[0][0:-10], phase3_b[1][0:-10], '-', marker='.', markersize=1.5, color='green', lw=lw1, label=r'q = 1.18')
lines += ax.plot(phase4_b[0][0:-10], phase4_b[1][0:-10], '-', marker='.', markersize=1.5, color=col2, lw=lw1, label=r'q = 1.57')
ax.labels = [l.get_label() for l in lines]
ax.legend(lines, ax.labels, bbox_to_anchor=(0.45, 0.8), fontsize=4.5, frameon=False)

ax.text(0.8*xxx, yy + 0.2, r'$\lambda$', size=ss)
ax.text(xx + 0.05, 0.85*yyy, r'$\mu$', size=ss)
ax.text(xxx - 0.1*(xxx-xx), yyy - 0.1*(yyy - yy), r'(d)', size=ssl)

ax.axis([xx, xxx, yy, yyy])




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

text(0.9*xxx-0.4, 0.9*yyy, r'(e)', size=ssl)
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

ax = axes([l22, b22, w22, h22])
ax.tick_params(axis='y', direction='in', length=lt, pad=p, width=w)
ax.tick_params(axis='x', direction='in', length=lt, pad=p, width=w)

xx  = 0
xxx = 1
yy  = 0
yyy = 1

ax.set_yticklabels([])
ax.set_xticklabels([])

ax.axis([xx, xxx, yy, yyy])

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

i_xx  = min(mean_0) - 0.35*(max(mean_0)-min(mean_0))
i_xxx = max(mean_0) + 0.15*(max(mean_0)-min(mean_0))
i_yy  = min(std_0) - 0.25*(max(std_0)-min(std_0))
i_yyy = max(std_0) + 0.15*(max(std_0)-min(std_0))

ax4.text(i_xxx - 0.6*(i_xxx - i_xx), i_yy + 0.08*(i_yyy - i_yy),  r'$\langle W_0 \rangle$', size=1.2*ratio*ss)
ax4.text(i_xx + 0.05*(i_xxx - i_xx), i_yyy - 0.3*(i_yyy - i_yy), r'$\sigma_0^2$', size=1.5*ratio*ss)

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

i_xx  = min(mean_1) - 0.35*(max(mean_1)-min(mean_1))
i_xxx = max(mean_1) + 0.15*(max(mean_1)-min(mean_1))
i_yy  = min(std_1) - 0.25*(max(std_1)-min(std_1))
i_yyy = max(std_1) + 0.15*(max(std_1)-min(std_1))

ax2.text(i_xxx - 0.6*(i_xxx - i_xx), i_yy + 0.08*(i_yyy - i_yy),  r'$\langle W_1 \rangle$', size=1.2*ratio*ss)
ax2.text(i_xx + 0.05*(i_xxx - i_xx), i_yyy - 0.3*(i_yyy - i_yy), r'$\sigma_1^2$', size=1.5*ratio*ss)
ax2.text(i_xxx - 0.15*(i_xxx - i_xx), i_yyy - 0.15*(i_yyy - i_yy), r'(f)', size=ssl)

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

i_xx  = min(mean_2) - 0.35*(max(mean_2)-min(mean_2))
i_xxx = max(mean_2) + 0.15*(max(mean_2)-min(mean_2))
i_yy  = min(std_2) - 0.25*(max(std_2)-min(std_2))
i_yyy = max(std_2) + 0.15*(max(std_2)-min(std_2))

ax5.text(i_xxx - 0.6*(i_xxx - i_xx), i_yy + 0.08*(i_yyy - i_yy),  r'$\langle W_7 \rangle$', size=1.2*ratio*ss)
ax5.text(i_xx + 0.05*(i_xxx - i_xx), i_yyy - 0.3*(i_yyy - i_yy), r'$\sigma_7^2$', size=1.5*ratio*ss)

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

i_xx  = min(mean_mult) - 0.33*(max(mean_mult)-min(mean_mult))
i_xxx = max(mean_mult) + 0.15*(max(mean_mult)-min(mean_mult))
i_yy  = min(std_mult) - 0.25*(max(std_mult)-min(std_mult))
i_yyy = max(std_mult) + 0.15*(max(std_mult)-min(std_mult))

ax3.text(i_xxx - 0.6*(i_xxx - i_xx), i_yy + 0.08*(i_yyy - i_yy),  r'$\langle W \rangle$', size=1.2*ratio*ss)
ax3.text(i_xx + 0.05*(i_xxx - i_xx), i_yyy - 0.2*(i_yyy - i_yy), r'$\sigma^2$', size=1.5*ratio*ss)

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
filename = 'fig_SI_poster.pdf'
savefig(filename, dpi=3e2)

