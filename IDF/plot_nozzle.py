# script for plotting quasi-1d nozzle flow Tecplot data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_nozzle(out_file, in_file='./quasi1d.dat'):
    # set some formating parameters
    axis_fs = 12 # axis title font size
    axis_lw = 1.0 # line width used for axis box, legend, and major ticks
    label_fs = 10 # axis labels' font size

    # get data to plot
    data = open(in_file, 'r')
    # data = open('./quasi1d.dat', 'r')
    [x, A, rho, rhou, e, p, p_targ, u, Mach, Mach_exact] = \
        np.loadtxt(data, skiprows=3, unpack=True)

    # set figure size in inches, and crete a single set of axes
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the data
    # ms = markersize
    # mfc = markerfacecolor
    # mew = markeredgewidth
    press, = ax.plot(x, p, '-k', linewidth=1.5)
    press_targ, = ax.plot(
        x, p_targ, '--ks', linewidth=0.5, ms=4.0, mfc='w', mew=1.5, markevery=1)

    # Tweak the appeareance of the axes
    ax.set_xlabel('x', fontsize=axis_fs, weight='bold')
    ax.set_ylabel('Pressure', fontsize=axis_fs, weight='bold', labelpad=11)
    ax.grid(which='major', axis='y')
    ax.axis([0.0, 1.0, 0.99*min(p_targ), 1.01*max(p_targ)])  # axes ranges

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
    ax.xaxis.set_ticks(np.arange(0, 1.0, 0.1), minor=True)
    ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)

    # Now add second axis if desired
    ax2 = ax.twinx()
    area, = ax2.plot(x, A, '--k', linewidth=1.0)
    ax2.set_ylabel('Area', fontsize=axis_fs, weight='bold', labelpad=11)
    ax2.axis([0.0, 1.0, 0.95*min(A), 1.05*max(A)])

    # plot and tweak the legend
    leg = ax.legend(
        (press, press_targ, area),
        ('pressure', 'targ. press.', 'nozzle area'),
        loc='upper right', numpoints=1,
        borderpad=0.75, handlelength=4)
    rect = leg.get_frame()
    rect.set_linewidth(axis_lw)
    for t in leg.get_texts():
        t.set_fontsize(axis_fs)    # the legend text fontsize

    plt.savefig(out_file)
    plt.close()
