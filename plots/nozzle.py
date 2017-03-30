import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

parser = argparse.ArgumentParser(description='Plot initial or final nozzle.')
parser.add_argument('--init', action='store_true')
parser.add_argument('--show', action='store_true')
args = parser.parse_args()

# set some formating parameters
axis_fs = 8 # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 8 # axis labels' font size

# get data to plot
file_dir = '../IDF/121node/20dv'
if args.init:
    in_file = '%s/init_pressure_area.dat'%file_dir
else:
    in_file = '%s/quasi1d.dat'%file_dir
data = open(in_file, 'r')
[x, A, rho, rhou, e, p, p_targ, u, Mach, Mach_exact] = \
    np.loadtxt(data, skiprows=3, unpack=True)

# set figure size in inches, and crete a single set of axes
fig = plt.figure(figsize=(4, 3), dpi=300)
ax = fig.add_subplot(111)

# plot the data
ms = 4.0 # marker size
mew = 0.75 # marker edge width
lw = 0.75 # line width
press, = ax.plot(x, p, '-k', linewidth=lw)
press_targ, = ax.plot(
    x, p_targ, ':ks', linewidth=2*lw, ms=ms, mfc='w', mew=mew, markevery=1)

# Tweak the appeareance of the axes
ax.set_xlabel('x', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Pressure', fontsize=axis_fs, weight='bold', labelpad=6)
ax.grid(which='major', axis='y')
ax.axis([0.0, 1.0, 0.99*min(p_targ), 1.01*max(p_targ)])  # axes ranges

# adjust gridlines
gridlines = ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle(':')

# ticks on bottom and left only
ax.xaxis.tick_bottom() # use ticks on bottom only
ax.yaxis.tick_left()
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

# define and format the minor ticks
ax.xaxis.set_ticks(np.arange(0, 1.0, 0.1), minor=True)
ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)

# Now add second axis if desired
ax2 = ax.twinx()
area, = ax2.plot(x, A, '--k', linewidth=lw)
ax2.set_ylabel('Area', fontsize=axis_fs, weight='bold', labelpad=12, rotation=270)
ax2.axis([0.0, 1.0, 0.95*1.49, 1.05*2.])
# ticks on bottom and left only
ax2.yaxis.tick_right()
for label in ax2.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

# plot and tweak the legend
ax.legend(
    (press, press_targ, area),
    ('Pressure', 'Targ. press.', 'Nozzle Area'),
    loc='upper right', numpoints=1, fontsize=label_fs, labelspacing=0.75,
    borderpad=0.75, handlelength=3, fancybox=False, framealpha=1.0, edgecolor='k')

save_dir = '/Users/denera/Documents/RPI/Optimal Design Lab/IDF-RSNK-journal'
if args.init:
    out_name = 'init_nozzle'
else:
    out_name = 'final_nozzle'
plt.savefig('%s/%s.eps'%(save_dir, out_name), format='eps', dpi=300,
            bbox_inches='tight')
plt.savefig('%s/%s.png'%(save_dir, out_name), format='png',
            bbox_inches='tight')
if args.show:
    plt.show()
plt.close()