"""
hcl_pdra25: plot.py

Contains plotting functions.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


""" Class to store information about a particular spectogram"""
class Spectogram:
    def __init__(self, x, y, z):
        self.x = x # x-axis, usually times
        self.y = y # y-axis, usually angles or energies
        self.z = z # z-axis, usually eV/cm^2 s sr eV


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
        legend_bbox = kwargs.get('legend_bbox', (0.965,0.5))
        ax.legend(kwargs['labels'], ncol=1, labelcolor='linecolor', labelspacing=0.15, handlelength=0.0, loc='center left', alignment='left', bbox_to_anchor=legend_bbox, frameon=False)
    
    # set up y-label
    if kwargs.get('ylabel', None) is not None:
        y_labelpad = kwargs.get('y_labelpad', 50)
        ax.set_ylabel(kwargs['ylabel'], labelpad=y_labelpad, ha='center', va='center')

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


""" Function to plot a feather on a given time series axis """
def plot_feather(ax, times, data, **kwargs):

    # skip every so many points
    skip = kwargs.get('skip',1)

    # retrieve which two components to get
    x_cmpt, y_cmpt = kwargs.get('components',[1,0])

    # using quiver to plot a feather plot
    ax.quiver(
        times[::skip], 
        np.zeros(times.shape)[::skip], 
        data[::skip,x_cmpt], 
        data[::skip,y_cmpt], 
        angles=kwargs.get('angles','uv'), 
        scale_units=kwargs.get('scale_units','y'), 
        scale=kwargs.get('scale',1), 
        width=kwargs.get('width',0.0005)
    )

    # set up legend
    if kwargs.get('labels', None) is not None:
        legend_bbox = kwargs.get('legend_bbox', (0.965,0.5))
        ax.legend(kwargs['labels'], ncol=1, labelcolor='linecolor', labelspacing=0.15, handlelength=0.0, loc='center left', alignment='left', bbox_to_anchor=legend_bbox, frameon=False)
    
    # set up y-label
    if kwargs.get('ylabel', None) is not None:
        y_labelpad = kwargs.get('y_labelpad', 50)
        ax.set_ylabel(kwargs['ylabel'], labelpad=y_labelpad, ha='center', va='center')

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
    yaxis = kwargs.get('yaxis', 'e_pad')
    
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

    # define the norm, min_max being the full range in Z
    if isinstance(kwargs.get('norm'), str):
        if kwargs.get('norm').lower() == 'min_max':
            Z_no_zeros = np.where(Z==0.0,np.nan,Z)
            norm = mpl.colors.LogNorm(vmin=np.nanmin(Z_no_zeros), vmax=np.nanmax(Z_no_zeros))
    else:
        norm = kwargs.get('norm', mpl.colors.LogNorm())

    pcm = ax.pcolormesh(X, Y, Z.T, rasterized=True, shading="flat", norm=norm, cmap=kwargs.get('cmap','turbo'))
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width +0.005, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(mappable=pcm, cax=cax, ax=ax, orientation="vertical")
    ax.set_ylabel(kwargs['ylabel'], labelpad=50, ha='center', va='center')
    cax.set_ylabel(kwargs['label'], fontsize=23)
    cax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=7))

    result = Spectogram(X, Y, Z)
    return result


""" Function to take and plot a 1D cut """
def PAD_1D_cut(PAD_channel, target_time: str, title: str = "", **kwargs):

    # index of closest time to target
    idx = np.argmin(np.abs(PAD_channel.x - np.datetime64(target_time)))

    # get the 1D slice at idx
    cut = PAD_channel.z[idx,:]

    # plot the 1D slice
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.step(PAD_channel.y, np.r_[cut, cut[-1]], where='post')

    # x axis
    ax.set_xlabel(r'$\mathrm{Angle}~[^\circ]$')
    ax.set_xlim(0,180)
    ax.set_xticks([0,45,90,135,180], labels=['$0$','$45$','$90$','$135$','$180$'])
    ax.set_xticks([15,30,60,75,105,120,150,165], minor=True)

    # y axis
    if not kwargs.get('y_lim', None) is None:
        ax.set_ylim(kwargs.get('y_lim')[0], kwargs.get('y_lim')[-1])
    ax.set_yscale('log')
    ax.set_ylabel(r'$\mathrm{Electron~PAD}$' '\n' r'$\mathrm{[eV/cm^2\,s\,sr\,eV]}$')

    # Plot format
    ax.set_title(f'{title} ({target_time})', fontsize=28, pad=20)
    plt.show()


""" Routine to plot B and v field feathers along an orbit trajectory"""
def plot_orbit_feather(scpos_us, scpos_ds, B_vec, v_vec, iskip_B=1, iskip_v=1, ax_vec=(), **kwargs):

    # get which axes idx to print
    which_options = [['xy','xz','yz'],[(0,1),(0,2),(1,2)]]
    which = kwargs.get('which', 'xy')
    assert which in which_options[0], f"Only {which_options[0]} are supported vector components."
    ax1, ax2 = which_options[-1][which_options[0].index(which)]

    fig = plt.figure(figsize=kwargs.get('figsize',(13,13)))
    ax = fig.add_subplot()

    ax.plot(scpos_us[:,ax1], scpos_us[:,ax2], label=kwargs.get('orbitlabel',None), color='purple')

    # Magnetic field feather
    ax.quiver(
        scpos_us[::iskip_B, ax1], 
        scpos_us[::iskip_B, ax2], 
        B_vec[::iskip_B, ax1], 
        B_vec[::iskip_B, ax2], 
        angles='uv', scale_units='dots', scale=10*kwargs.get('scale_multiplier',1), width=0.0025
    )

    # velocity field feather
    if kwargs.get('plot_velocity', False):
        ax.quiver(
            scpos_ds[::iskip_v, ax1], 
            scpos_ds[::iskip_v, ax2], 
            v_vec[::iskip_v,ax1], 
            v_vec[::iskip_v,ax2], 
            1,
            alpha=0.5,
            angles='uv', scale_units='dots', scale=5*kwargs.get('scale_multiplier',1), width=0.0025,
            cmap='bwr', norm=mpl.colors.Normalize(vmin=-1, vmax=1)
        )

    ### PLOT LMN ARROWS ###
    if not ax_vec == ():
        X_vec, Y_vec, Z_vec = ax_vec
        zero_loc = kwargs.get('eg_axis_loc', [0,0])

        # X arrow, red
        ax.quiver(
            zero_loc[0], 
            zero_loc[1], 
            X_vec[ax1], 
            X_vec[ax2], 
            1,
            angles='uv', scale_units='inches', units='inches', scale=1, width=0.03, zorder=50,
            cmap='bwr', norm=mpl.colors.Normalize(vmin=-1, vmax=1)
        )

        # Y arrow, blue
        ax.quiver( 
            zero_loc[0], 
            zero_loc[1], 
            Y_vec[ax1], 
            Y_vec[ax2], 
            -1,
            angles='uv', scale_units='inches', units='inches', scale=1, width=0.03, zorder=50,
            cmap='bwr', norm=mpl.colors.Normalize(vmin=-1, vmax=1)
        )

        # z arrow, green
        ax.quiver(
            zero_loc[0], 
            zero_loc[1], 
            Z_vec[ax1], 
            Z_vec[ax2], 
            1,
            angles='uv', scale_units='inches', units='inches', scale=1, width=0.03, zorder=50,
            cmap='PiYG', norm=mpl.colors.Normalize(vmin=-1, vmax=1)
        )

    # Plot any custom scatter point markers
    if kwargs.get('custom_scatter', None) is not None:
        for key, item in kwargs.get('custom_scatter').items():
            ax.scatter(scpos_ds[item['idx'], ax1], scpos_ds[item['idx'], ax2], marker=item['marker'], s=item['s'], fc=item['fc'], ec=item['ec'], zorder=100, label=item['label'])

    if kwargs.get('flip_x_axis', False):
        xlim_before_flip = ax.get_xlim()
        ax.set_xlim(xlim_before_flip[-1], xlim_before_flip[0])
    if kwargs.get('flip_y_axis', False):
        ylim_before_flip = ax.get_ylim()
        ax.set_ylim(ylim_before_flip[-1], ylim_before_flip[0])
    if kwargs.get('stretch_x_axis', 1) != 1:
        x_stretch_factor = kwargs.get('stretch_x_axis', 1)
        xlim_before_stretch = ax.get_xlim()
        xlim_midpoint = np.mean(xlim_before_stretch)
        xlim_range = xlim_before_stretch[-1] - xlim_before_stretch[0]
        ax.set_xlim(xlim_midpoint - (xlim_range/2)*x_stretch_factor, xlim_midpoint + (xlim_range/2)*x_stretch_factor)
    ax.set_xlabel(kwargs.get('xlabel',None))
    ax.set_ylabel(kwargs.get('ylabel',None))
    ax.set_title(kwargs.get('title', None))
    ax.grid()
    ax.legend(loc='lower right', ncol=1, fontsize=14)
    if kwargs.get('equal', True):
        ax.set_aspect('equal')

    plt.show()


""" Routine to estimate PAD bin edges from bin centers"""
def get_bin_edges(centers):

    # For readability
    log_bins = np.log10(centers)
    log_bins_diff = np.diff(log_bins)

    # Assume even spacing in log-space
    log_bin_edges = np.r_[log_bins[0]-log_bins_diff[0]/2, np.add(log_bins[:-1], log_bins_diff/2), log_bins[-1]+log_bins_diff[-1]/2]

    # 10^x
    bin_edges = np.power(10, log_bin_edges)

    return bin_edges