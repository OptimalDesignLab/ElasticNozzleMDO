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
    fig = plt.figure(figsize=(7, 4), facecolor='w')
    ax = fig.add_subplot(111)

    # plot the data
    # ms = markersize
    # mfc = markerfacecolor
    # mew = markeredgewidth
    press, = ax.plot(x, p, '-k', linewidth=1.5)
    press_targ, = ax.plot(
        x, p_targ, '--ks', linewidth=0.5, ms=4.0, mfc='w', mew=1.5, markevery=1)
    # qn = ax.plot(
    #     ndv, qn/flow_cost, '-k^', linewidth=3.0, ms=8.0, mfc='w', mew=1.5,
    #     color=(0.35, 0.35, 0.35))

    # Tweak the appeareance of the axes
    ax.set_position([0.12, 0.12, 0.78, 0.83]) # position relative to figure edge
    ax.set_xlabel('x', fontsize=axis_fs, weight='bold')
    ax.set_ylabel(
        'pressure', fontsize=axis_fs, weight='bold', labelpad=11)
    ax.grid(which='major', axis='y', linestyle='--')
    ax.axis([0.0, 1.0, 0.99*min(p), 1.01*max(p)])  # axes ranges
    rect = ax.patch # a Rectangle instance
    # rect.set_facecolor('white')
    # rect.set_ls('dashed')
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
    ax.xaxis.set_ticks(np.arange(0, 1.0, 0.1), minor=True)
    ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
    # ax.yaxis.set_ticks(np.arange(10,300,10),minor=True)
    # ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
    # print ax.xaxis.get_ticklines(minor=True)

    # Now add second axis if desired
    ax2 = ax.twinx()
    area, = ax2.plot(x, A, '--r', linewidth=1.0)
    ax2.set_ylabel(
        'area', fontsize=axis_fs, weight='bold', labelpad=11)
    ax2.axis([0.0, 1.0, 0.95*min(A), 1.05*max(A)])

    # turn off tick on right and upper edges; this is now down above
    # for tick in ax.xaxis.get_major_ticks():
    #    tick.tick2On = False
    # for tick in ax.yaxis.get_major_ticks():
    #    tick.tick2On = False

    # plot and tweak the legend
    leg = ax.legend(
        (press, press_targ, area),
        ('pressure', 'targ. press.', 'nozzle area'),
        loc=(0.6, 0.65), numpoints=1,
        borderpad=0.75, handlelength=4)
    rect = leg.get_frame()
    rect.set_linewidth(axis_lw)
    for t in leg.get_texts():
        t.set_fontsize(axis_fs)    # the legend text fontsize

    plt.savefig(out_file)
    plt.close()
