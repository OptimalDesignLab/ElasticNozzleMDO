import numpy as np
import matplotlib.pyplot as plt

# folder names
idf_folder = '../IDF/121node'
mdf_folder = '../MDF/trust_121node'

# loop over test cases and collect cost information
dv = 16
flow_cost = 83.
idf_file = '%s/%idv/kona_hist.dat'%(idf_folder, dv)
[idf_cost, idf_opt, idf_feas] = np.loadtxt(idf_file, usecols=[1, 2, 3], unpack=True)
mdf_file = '%s/%idv/kona_hist.dat'%(mdf_folder, dv)
[mdf_cost, mdf_opt] = np.loadtxt(mdf_file, usecols=[1, 2], unpack=True)

# set some formating parameters
axis_fs = 12 # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 10 # axis labels' font size

# set figure size in inches, and crete a single set of axes
fig = plt.figure()
ax = fig.add_subplot(111)

# plot the data
# ms = markersize
# mfc = markerfacecolor
# mew = markeredgewidth
# mec = markeredgecolor
ax.semilogy(idf_cost/flow_cost, idf_opt/idf_opt[0], '-k^', 
            linewidth=1.5, ms=8.0, mfc='w', mew=1.5, mec='k', label='IDF Optimality')
ax.semilogy(idf_cost/flow_cost, idf_feas/idf_feas[0], '-kx', 
            linewidth=1.5, ms=8.0, mfc='w', mew=1.5, mec='k', label='IDF Feasibility')
ax.semilogy(mdf_cost/flow_cost, mdf_opt/mdf_opt[0], '--ko', 
            linewidth=1.5, ms=8.0, mfc='w', mew=1.5, mec='k', label='MDF Optimality')

# Tweak the appeareance of the axes
ax.axis([0, 60, 10**-7, 10**1])  # axes ranges
ax.set_xlabel('Cost (equivalent MDA solutions)', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Relative Norms', fontsize=axis_fs, weight='bold', labelpad=12)

# ticks on bottom and left only
ax.xaxis.tick_bottom() # use ticks on bottom only
ax.yaxis.tick_left()
for line in ax.xaxis.get_ticklines():
    line.set_markersize(5) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for line in ax.yaxis.get_ticklines():
    line.set_markersize(5) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

# plot and tweak the legend
leg = ax.legend(loc='lower left', numpoints=1, borderpad=0.75, handlelength=4)
rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(12)    # the legend text fontsize

plt.show()
