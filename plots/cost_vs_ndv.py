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
ndvs = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
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

# compute cost difference and % increase
avg_idf = sum(idf_data/flow_cost)/len(idf_data)
avg_mdf = sum(mdf_data/flow_cost)/len(mdf_data)
print "Average IDF cost = %f"%avg_idf
print "Average MDF cost = %f"%avg_mdf
cost_diff = idf_data/flow_cost - mdf_data/flow_cost
pct_diff = 100.*cost_diff/(mdf_data/flow_cost)
print "Cost differences = %r"%cost_diff
print "Percent diffs    = %r"%pct_diff
avg_diff = avg_idf - avg_mdf
avg_pct = 100.*avg_diff/avg_mdf
print "Average cost difference = %f"%avg_diff
print "Average pct difference  = %f"%avg_pct
# exit()

# set figure size in inches, and crete a single set of axes
fig = plt.figure(figsize=(4, 3), dpi=300)
ax = fig.add_subplot(111)

# set some formating parameters
axis_fs = 8 # axis title font size
label_fs = 8 # axis labels' font size

# plot the data
ms = 4.0 # marker size
mew = 0.75 # marker edge width
lw = 0.75 # line width
idf = ax.plot(ndvs, idf_data/flow_cost, '-k^', linewidth=lw, 
              ms=ms, mfc='w', mew=mew, mec='k', label='IDF')
mdf = ax.plot(ndvs, mdf_data/flow_cost, ':ko', linewidth=2*lw, 
              ms=ms, mfc='w', mew=mew, mec='k', label='MDF')

# Tweak the appeareance of the axes
ax.axis([5, 45, 0, 90])
ax.set_xlabel('Number of design variables', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Cost (equivalent MDA solutions)', fontsize=axis_fs, weight='bold', labelpad=6)

# ticks on bottom and left only
ax.xaxis.tick_bottom() # use ticks on bottom only
ax.yaxis.tick_left()
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

# plot and tweak the legend
leg = ax.legend(loc='best', fontsize=label_fs, labelspacing=0.75, borderpad=0.75, 
                numpoints=1, handlelength=3, fancybox=False, framealpha=1.0, edgecolor='k')

save_dir = '/Users/denera/Documents/RPI/Optimal Design Lab/IDF-RSNK-journal'
plt.savefig('%s/nozzle_costcomp.eps'%save_dir, format='eps', dpi=300,
            bbox_inches='tight')
plt.savefig('%s/nozzle_costcomp.png'%save_dir, format='png',
            bbox_inches='tight')
if args.show:
    plt.show()
plt.close()
