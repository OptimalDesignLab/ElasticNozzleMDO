# script for plotting quasi-1d nozzle flow Tecplot data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

def plot_nozzle(out_file, in_file='./quasi1d.dat'):
    # set some formating parameters
    axis_fs = 8 # axis title font size
    axis_lw = 1.0 # line width used for axis box, legend, and major ticks
    label_fs = 8 # axis labels' font size

    # get data to plot
    data = open(in_file, 'r')
    # data = open('./quasi1d.dat', 'r')
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
        x, p_targ, ':ks', linewidth=0.5*lw, ms=ms, mfc='w', mew=mew, markevery=1)

    # Tweak the appeareance of the axes
    ax.set_xlabel('x', fontsize=axis_fs, weight='bold')
    ax.set_ylabel('Pressure', fontsize=axis_fs, weight='bold', labelpad=6)
    ax.grid(which='major', axis='y')
    ax.axis([0.0, 1.0, 0.99*min(p_targ), 1.01*max(p_targ)])  # axes ranges

    # define and format the minor ticks
    ax.xaxis.set_ticks(np.arange(0, 1.0, 0.1), minor=True)
    ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)

    # Now add second axis if desired
    ax2 = ax.twinx()
    area = ax2.plot(x, A, '--k', linewidth=lw)
    ax2.set_ylabel('Area', fontsize=axis_fs, weight='bold', labelpad=6)
    ax2.yaxis.set_tick_params(which='major', labelsize=label_fs)
    ax2.axis([0.0, 1.0, 0.95*1.49, 1.05*2.])

    # plot and tweak the legend
    ax.legend(
        (press, press_targ, area),
        ('pressure', 'targ. press.', 'nozzle area'),
        loc='upper right', numpoints=1, fontsize=label_fs, labelspacing=0.75,
        borderpad=0.75, handlelength=3)

    plt.savefig(out_file)
    plt.close()
