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

# folder names
idf_folder = '../IDF/121node'
mdf_folder = '../MDF/trust_121node'

# loop over test cases and collect cost information
dv = 20
flow_cost = 83.
idf_file = '%s/%idv/kona_hist.dat'%(idf_folder, dv)
[idf_cost, idf_opt, idf_feas] = np.loadtxt(idf_file, usecols=[1, 2, 3], unpack=True)
mdf_file = '%s/%idv/kona_hist.dat'%(mdf_folder, dv)
[mdf_cost, mdf_opt] = np.loadtxt(mdf_file, usecols=[1, 2], unpack=True)

# set some formating parameters
axis_fs = 8 # axis title font size
label_fs = 8 # axis labels' font size

# set figure size in inches, and crete a single set of axes
fig = plt.figure(figsize=(4, 3), dpi=300)
ax = fig.add_subplot(111)

# plot the data
ms = 4.0 # marker size
mew = 0.75 # marker edge width
lw = 0.75 # line width
ax.semilogy(idf_cost/flow_cost, idf_opt/idf_opt[0], '-k^', 
            linewidth=lw, ms=ms, mfc='w', mew=mew, mec='k', label='IDF Optimality')
ax.semilogy(idf_cost/flow_cost, idf_feas/idf_feas[0], '-kx', 
            linewidth=lw, ms=ms, mfc='w', mew=mew, mec='k', label='IDF Feasibility')
ax.semilogy(mdf_cost/flow_cost, mdf_opt/mdf_opt[0], ':ko', 
            linewidth=2*lw, ms=ms, mfc='w', mew=mew, mec='k', label='MDF Optimality')

# Tweak the appeareance of the axes
ax.axis([0, 60, 10**-6, 10**1])  # axes ranges
ax.set_xlabel('Cost (equivalent MDA solutions)', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Relative Norms', fontsize=axis_fs, weight='bold', labelpad=6)

# ticks on bottom and left only
ax.xaxis.tick_bottom() # use ticks on bottom only
ax.yaxis.tick_left()
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

# plot and tweak the legend
leg = ax.legend(loc='lower left', fontsize=label_fs, labelspacing=0.75, borderpad=0.75, 
                numpoints=1, handlelength=3, fancybox=False, framealpha=1.0, edgecolor='k')

save_dir = '/Users/denera/Documents/RPI/Optimal Design Lab/IDF-RSNK-journal'
plt.savefig('%s/nozzle_convergence.eps'%save_dir, format='eps', dpi=300,
            bbox_inches='tight')
plt.savefig('%s/nozzle_convergence.png'%save_dir, format='png',
            bbox_inches='tight')
if args.show:
    plt.show()
plt.close()
