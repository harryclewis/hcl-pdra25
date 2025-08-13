"""
hcl_pdra25: plot.py

Contains plotting functions.
"""

import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np

""" Function to plot a timeseries on a given axis """
def plot_ts(ax, var, **kwargs):

    assert not (isinstance(kwargs.get('multi_line', False), list) and kwargs.get('multi_line', False) is False), "To pass a list of variables, set multi_line=True"

    # plotting multiple lines on one axis/legend combo
    if kwargs.get('multi_line', False):
        for e,i in enumerate(var):
            if kwargs.get('c', None) is not None:
                ax.plot(i.time, i, c=kwargs['c'][e]) # index the color kwarg if multiple lines
            else:
                ax.plot(i.time, i)
    # plotting single line, or magnitude of vector object
    else:
        if kwargs.get('plot_mag', False):
            ax.plot(var.time, np.linalg.norm(var.values, axis=1), c='black')
        ax.plot(var.time, var, c=kwargs.get('c', None))

    # set up legend
    if kwargs.get('labels', None) is not None:
        ax.legend(kwargs['labels'], ncol=1, labelcolor='linecolor', labelspacing=0.15, handlelength=0.0, loc='center left', alignment='left', bbox_to_anchor=(0.965,0.5), frameon=False)
    
    # set up y-label
    if kwargs.get('ylabel', None) is not None:
        ax.set_ylabel(kwargs['ylabel'], labelpad=50, ha='center', va='center')

    # set up y-limits
    if kwargs.get('ylim', None) is not None:
        ax.set_ylim(kwargs['ylim'])

    # y-tick locators
    if kwargs.get('y_log', False):
        ax.yaxis.set_major_locator(mticker.LogLocator())
        ax.set_yscale('log')
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    # text to show that values are scaled (e.g. x10^10)
    if kwargs.get('scaling_text', None) is not None:
        ax.text(0.01, 0.97, kwargs['scaling_text'], transform=ax.transAxes, ha='left', va='top')

    # add a line at x-axis zero
    if kwargs.get('zero_line', False):
        ax.axhline(0.0, c='black', zorder=0)

    # add text, either in default location or specified in the kwarg
    if kwargs.get('text', None) is not None:
        txt = kwargs['text']
        if isinstance(txt, str):
            t = ax.text(1.00, 1.00, txt, transform=ax.transAxes, ha='right', va='center', fontsize=20)
            t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black'))
        else:
            ax.text(txt[1], txt[2], txt[0], transform=ax.transAxes, ha=txt[3], va=txt[4], fontsize=20)

    # add a horizontal line if requested
    if kwargs.get('y_line', None) is not None:
        ax.axhline(kwargs['y_line'], c='black',ls='dashed')


""" Function to plot a spectogram on a given axis """
def plot_spectr(fig, ax, var, **kwargs):

    # shift the times to match the center of the data points
    times = var.time
    dt = np.median(np.sort(np.diff(times)))
    times_shifted = times - dt / 2 
    X = np.concatenate((times_shifted, np.array([times[-1] + dt/2])))
    yaxis = kwargs.get('yaxis', None)
    
    if yaxis.lower() in ['e_spectro','i_spectro']:
        # plot spectrogram
        ax.set_yscale('log')
        lc = np.log10(kwargs.get('centers', None))
        Y = (10**np.r_[[lc[0]-(lc[1]-lc[0])/2], (lc[0:-1]+lc[1:])/2, [lc[-1]+(lc[-1]-lc[-2])/2]])
        Z = var.values

    elif yaxis.lower() == 'e_pad':
        # plot a pitch angle distribution
        Y = np.linspace(0,180,13)
        bins = var.axes[2].values[0]
        if kwargs.get('bin_idx', None) is None:
            # no energy limit, or energy range specified
            E_lim = kwargs.get('E_lim', [300, 800])
            inds = np.squeeze(np.argwhere(np.where((bins>=E_lim[0]) & (bins<=E_lim[-1]), 1, 0)))
            Z = np.nanmean(var.values[:,:,inds[0]:inds[-1]+1], axis=-1)
        else:
            # energy bin indices specified
            Z =var.values[:,:,kwargs.get('bin_idx')]

        ax.set_yticks([0,45,90,135,180], labels=['$0$','$45$','$90$','$135$','$180$'])
        ax.set_yticks([15,30,60,75,105,120,150,165], minor=True)

    pcm = ax.pcolormesh(X, Y, Z.T, rasterized=True, shading="auto",norm=kwargs.get('norm', mpl.colors.LogNorm()), cmap='turbo')
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width +0.005, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(mappable=pcm, cax=cax, ax=ax, orientation="vertical")
    ax.set_ylabel(kwargs['ylabel'], labelpad=50, ha='center', va='center')
    cax.set_ylabel(kwargs['label'], fontsize=23)
    cax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=7))
    return Z