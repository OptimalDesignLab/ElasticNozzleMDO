import numpy as np
import matplotlib.pyplot as plt

# folder names
idf_folder = '../IDF/121node'
mdf_folder = '../MDF/trust_121node'

# loop over test cases and collect cost information
ndvs = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
idf_data = np.zeros(len(ndvs))
mdf_data = np.zeros(len(ndvs))
flow_cost = 83.
for i, dv in enumerate(ndvs):
    idf_file = '%s/%idv/kona_hist.dat'%(idf_folder, dv)
    idf_cost = np.loadtxt(idf_file, usecols=[1], unpack=True)
    idf_data[i] = idf_cost[-1]
    mdf_file = '%s/%idv/kona_hist.dat'%(mdf_folder, dv)
    mdf_cost = np.loadtxt(mdf_file, usecols=[1], unpack=True)
    mdf_data[i] = mdf_cost[-1]

# set figure size in inches, and crete a single set of axes
fig = plt.figure()
ax = fig.add_subplot(111)

# set some formating parameters
axis_fs = 12 # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 10 # axis labels' font size

# plot the data
# ms = markersize
# mfc = markerfacecolor
# mew = markeredgewidth
# mec = markeredgecolor
idf = ax.plot(ndvs, idf_data/flow_cost, '-k^', linewidth=1.5, 
              ms=8.0, mfc='w', mew=1.5, mec='k', label='IDF')
mdf = ax.plot(ndvs, mdf_data/flow_cost, '--ko', linewidth=1.5, 
              ms=8.0, mfc='w', mew=1.5, mec='k', label='MDF')

# Tweak the appeareance of the axes
ax.axis([5, 35, 0, 90])
ax.set_xlabel('Number of design variables', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Cost (equivalent MDA solutions)', fontsize=axis_fs, weight='bold', labelpad=12)

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
leg = ax.legend(loc='best', numpoints=1, borderpad=0.75, handlelength=4)
rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(12)    # the legend text fontsize

plt.show()
