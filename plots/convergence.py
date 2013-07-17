import numpy as np
import matplotlib.pyplot as plt

# set some formating parameters
axis_fs = 12 # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 10 # axis labels' font size

# get data to plot
data_bfgs = open('./BFGS_nnp120_nd20_hist.dat', 'r')
[bfgs_cost, bfgs_grad] = np.loadtxt(data_bfgs, skiprows=3, usecols=(3,4), unpack=True)
data_rsnk = open('./RSNK_nnp120_nd20_hist.dat', 'r')
[rsnk_cost, rsnk_grad] = np.loadtxt(data_rsnk, skiprows=3, usecols=(3,4), unpack=True)
flow_cost = 42

# set figure size in inches, and crete a single set of axes
fig = plt.figure(figsize=(7,4), facecolor='w')
ax = fig.add_subplot(111)

# plot the data
# ms = markersize
# mfc = markerfacecolor
# mew = markeredgewidth
bfgs = ax.plot(bfgs_cost/flow_cost, bfgs_grad, '-k^', linewidth=2.0, ms=8.0, mfc='w', mew=1.5, \
         color=(0.35, 0.35, 0.35), mec=(0.35, 0.35, 0.35))
rsnk = ax.plot(rsnk_cost/flow_cost, rsnk_grad, '--ko', linewidth=1.5, ms=8.0, \
              mfc=(0.35,0.35,0.35), mew=1.5, mec='k')

# Tweak the appeareance of the axes
ax.axis([0, 175, 10**-11, 10**-4])  # axes ranges
ax.set_position([0.12, 0.13, 0.86, 0.83]) # position relative to figure edges
ax.set_xlabel('Cost (equivalent MDA solutions)', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Design Norm', fontsize=axis_fs, weight='bold', \
              labelpad=12)
ax.set_yscale('log')
ax.grid(which='major', axis='y', linestyle='--')
ax.set_axisbelow(True) # grid lines are plotted below
rect = ax.patch # a Rectangle instance
#rect.set_facecolor('white')
#rect.set_ls('dashed')
rect.set_linewidth(axis_lw)
rect.set_edgecolor('k')

# ticks on bottom and left only
ax.xaxis.tick_bottom() # use ticks on bottom only
ax.yaxis.tick_left()
for line in ax.xaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for line in ax.yaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

# define and format the minor ticks
ax.xaxis.set_ticks(np.arange(0,175,20),minor=True)
ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
    #print ax.xaxis.get_ticklines(minor=True)

# turn off tick on right and upper edges; this is now down above
#for tick in ax.xaxis.get_major_ticks():
#    tick.tick2On = False
#for tick in ax.yaxis.get_major_ticks():
#    tick.tick2On = False

# plot and tweak the legend
leg = ax.legend(('BFGS', 'RSNK'), loc=(0.65,0.68), numpoints=1, borderpad=0.75, \
                handlelength=4) # handlelength controls the width of the legend
rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(12)    # the legend text fontsize

plt.show()
