import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

# read the data from the files
[pre_iters, pre_res, pre_grad, pre_feas, _, _, _, _] = \
    np.loadtxt('./kona_krylov_pre.dat', unpack=True)
[nopre_iters, nopre_res, nopre_grad, nopre_feas, _, _, _, _] = \
    np.loadtxt('./kona_krylov.dat', unpack=True)

# set some formating parameters
axis_fs = 8 # axis title font size
label_fs = 8 # axis labels' font size

# plot cconvergence
fig = plt.figure(figsize=(4, 3), dpi=300)
ax = fig.add_subplot(111)

ms = 4.0 # marker size
mew = 0.75 # marker edge width
lw = 0.75 # line width
ax.semilogy(pre_iters, pre_res, '-^k', ms=ms, mfc='w', mew=mew, 
            linewidth=lw, label='Preconditioned')
ax.semilogy(nopre_iters, nopre_res, ':ok', ms=ms, mfc='w', mew=mew, 
            linewidth=2*lw, label='Not Preconditioned')

# plot formatting
ax.axis([0, 16, 10**-4, 10**1])
ax.set_xlabel('Krylov iterations', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Relative residual norm', fontsize=axis_fs, weight='bold', labelpad=6)
ax.xaxis.tick_bottom() # use ticks on bottom only
ax.yaxis.tick_left()
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

leg = ax.legend(loc='upper right', fontsize=label_fs, labelspacing=0.75, borderpad=0.75, 
                numpoints=1, handlelength=3, fancybox=False, framealpha=1.0, edgecolor='k')

save_dir = '/Users/denera/Documents/RPI/Optimal Design Lab/IDF-RSNK-journal'
plt.savefig('%s/nozzle_krylov.eps'%save_dir, format='eps', dpi=300,
            bbox_inches='tight')
plt.savefig('%s/nozzle_krylov.png'%save_dir, format='png',
            bbox_inches='tight')
if args.show:
    plt.show()
plt.close()
