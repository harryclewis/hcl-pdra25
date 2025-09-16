"""
hcl_pdra25: animate.py

Contains animation functions.
"""

# 3rd party
import astropy.units as u
from astropy.visualization import ImageNormalize, PowerStretch
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# 1st party
from .data import CustomVariable


"""
    Function to animate a sequence of WISPR maps.
    Also projects a given coordinate system (deisnged for LMN) onto the sequence.
    Plots these coordinates and a line extending along (currently) the Y direction.
"""
from matplotlib.animation import FuncAnimation
def create_wispr_video(sequence, zero_coord, X_coord, Y_coord, Z_coord, dt=1000/5, dy=2):
    fig = plt.figure(figsize=(13, 8))
    global ax
    ax = fig.add_subplot(projection=sequence[0])

    wispr_norm = ImageNormalize(vmin=0, vmax=0.5e-11, stretch=PowerStretch(1/2.2))

    # Function run on every frame
    def animate(t):

        # kill the previous axis
        global ax
        ax.remove() 
        ax = fig.add_subplot(projection=sequence[t])

        # set the coordinates to degrees
        lon, lat = ax.coords
        lon.set_major_formatter('d')
        lat.set_major_formatter('d')

        # Plot the WISPR map
        sequence[t].plot(axes=ax, norm=wispr_norm, cmap='viridis')
        sequence[t].draw_limb(axes=ax, color='w')

        # Sort out the grid
        s_grid = sequence[t].draw_grid(axes=ax, color='w')
        s_grid['lat'].set_ticks([-60,-30,0,30,60] * u.deg)
        s_grid['lon'].set_ticks([0,30,60,90,120,150,180,210,240,270,300,330] * u.deg)

        # Get PSP's location (currently unused)
        # psp_hpc = psp.transform_to(sequence[t].coordinate_frame)

        # convert the sky coordinates to the instantaneous coordinate frame
        X_hcs = X_coord.transform_to(sequence[t].coordinate_frame)
        Y_hcs = Y_coord.transform_to(sequence[t].coordinate_frame)
        Z_hcs = Z_coord.transform_to(sequence[t].coordinate_frame)
        zero_hcs = zero_coord.transform_to(sequence[t].coordinate_frame)

        # wrap around the longitudes to prevent weirdness
        X_hcs_lon = X_hcs.data.lon.value*57.29583337126-360 if X_hcs.data.lon.value*57.29583337126 > 180 else X_hcs.data.lon.value*57.29583337126
        Y_hcs_lon = Y_hcs.data.lon.value*57.29583337126-360 if Y_hcs.data.lon.value*57.29583337126 > 180 else Y_hcs.data.lon.value*57.29583337126
        Z_hcs_lon = Z_hcs.data.lon.value*57.29583337126-360 if Z_hcs.data.lon.value*57.29583337126 > 180 else Z_hcs.data.lon.value*57.29583337126

        # Plot the X axis arrow
        ax.quiver(
            [zero_hcs.data.lon.value*57.29583337126],
            [zero_hcs.data.lat.value*57.29583337126],
            [X_hcs_lon],
            [X_hcs.data.lat.value*57.29583337126], 
            transform=ax.get_transform(frame=sequence[t].coordinate_frame), 
            angles='xy', 
            scale_units='xy', 
            scale=0.08, 
            width=0.003, 
            color='#0071b2', 
            zorder=15
        )
        # Plot the Y axis arrow
        ax.quiver(
            [zero_hcs.data.lon.value*57.29583337126],
            [zero_hcs.data.lat.value*57.29583337126],
            [Y_hcs_lon],
            [Y_hcs.data.lat.value*57.29583337126],
            transform=ax.get_transform(frame=sequence[t].coordinate_frame), 
            angles='xy', 
            scale_units='xy', 
            scale=0.08, 
            width=0.003, 
            color='#009e73', 
            zorder=15
        )
        # Plot the Z axis arrow
        ax.quiver(
            [zero_hcs.data.lon.value*57.29583337126],
            [zero_hcs.data.lat.value*57.29583337126],
            [Z_hcs_lon],
            [Z_hcs.data.lat.value*57.29583337126],
            transform=ax.get_transform(frame=sequence[t].coordinate_frame), 
            angles='xy', 
            scale_units='xy', 
            scale=0.08, 
            width=0.003, 
            color='#d55c00', 
            zorder=15
        )
                
        # Work out the gradient of the Y axis, the intercept, then the angle
        m = (Y_hcs.data.lat.value*57.29583337126 - zero_hcs.data.lat.value*57.29583337126) / (Y_hcs_lon - zero_hcs.data.lon.value*57.29583337126)
        c = Y_hcs.data.lat.value*57.29583337126 - (m * Y_hcs_lon)
        theta = np.atan2((Y_hcs.data.lat.value*57.29583337126 - zero_hcs.data.lat.value*57.29583337126), (Y_hcs_lon - zero_hcs.data.lon.value*57.29583337126))
        H = dy * np.cos(theta)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Plot the 'lower' boundary line (Y-dy)
        ax.plot(
            (np.linspace(-40, 0, 100)-c)/m + H*np.sin(-theta), 
            np.linspace(-40, 0, 100) + H*np.cos(-theta), 
            c='#f0e442', 
            lw=1, 
            transform=ax.get_transform(frame=sequence[t].coordinate_frame)
        )
        # Plot the Y-axis extension
        ax.plot(
            (np.linspace(-40, 0, 100)-c)/m, 
            np.linspace(-40, 0, 100), 
            c='white', 
            lw=1, 
            transform=ax.get_transform(frame=sequence[t].coordinate_frame)
        )
        # Plot the 'upper' boundary line (Y+dy)
        ax.plot(
            (np.linspace(-40, 0, 100)-c)/m - H*np.sin(-theta), 
            np.linspace(-40, 0, 100) - H*np.cos(-theta), 
            c='#56b3e9', 
            lw=1, 
            transform=ax.get_transform(frame=sequence[t].coordinate_frame)
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


    anim = FuncAnimation(
        fig,
        animate,
        frames=len(sequence),
        interval=dt
    )

    return anim


"""
    Function to animate a sequence of Solar Orbiter EUI maps.
    
    Basic plot without any additionals: simply SolO's view.
"""
def create_EUI_video(sequence, dt=1000/5):
    fig = plt.figure(figsize=(12,12))
    global ax
    ax = fig.add_subplot(projection=sequence[0])

    def animate(t):
        global ax
        ax.remove()
        ax = fig.add_subplot(projection=sequence[t])
        ax.set_xlim(0.15*3600,0.7*3600)
        ax.set_ylim(0.15*3600,0.7*3600)
        lon, lat = ax.coords
        lon.set_major_formatter('d')
        lat.set_major_formatter('d')
        sequence[t].plot(axes=ax, cmap='viridis')

    anim = FuncAnimation(
        fig,
        animate,
        frames=len(sequence),
        interval=dt
    )

    return anim


""" Routine to animate electron PAD with B_lmn timeseries above """
def animate_PAD(b_lmn_time, b_lmn_data, pad, bins, **kwargs):
    fig = plt.figure(figsize=(15,10))
    ax_ts = fig.add_subplot(6,1,1)
    plot_ts(ax_ts, CustomVariable(b_lmn_time, b_lmn_data), labels=[r'$B_L$',r'$B_M$',r'$B_N$'], ylabel=r'$\mathbf{B}_\mathrm{LMN}$' '\n' r'$\mathrm{[nT]}$', zero_line=True)
    majorlocator = mdates.AutoDateLocator(minticks=3, maxticks=12)
    minorlocator = mdates.MinuteLocator(byminute=np.arange(0,60,10))
    formatter = mdates.ConciseDateFormatter(majorlocator, usetex=True, zero_formats=['', '%Y', '%b', '%-d %b', '%H:%M', '%H:%M'])
    ax_ts.xaxis.set_major_locator(majorlocator)
    ax_ts.xaxis.set_minor_locator(minorlocator)
    ax_ts.xaxis.set_major_formatter(formatter)
    ax_ts.set_xlim(pd.to_datetime(t_start), pd.to_datetime(t_end))

    ax_polar = fig.add_subplot(6,1,(2,6), projection='polar')

    e_bin_edges = get_bin_edges(bins)
    bottom_bin, top_bin = kwargs.get('binrange', (31, 0))

    # Set up the polar plot
    ax_polar.set_thetamin(0)
    ax_polar.set_thetamax(180)
    ax_polar.set_rmin(bins[bottom_bin]*0.9)
    ax_polar.set_rmax(bins[top_bin]*1.1)
    ax_polar.set_rscale('log')
    ax_polar.set_position(pos=[0,-0.25,1,1.1])

    # Set the colour-axis range
    if not kwargs.get('vlim', None) is None:
        vmin, vmax = kwargs.get('vlim')
    else:
        vmin = np.nanmin(pad.values[pad.values>0.0])
        vmax = np.nanmax(pad.values)

    # Create the animated objects
    pc = ax_polar.pcolormesh(angles, e_bin_edges, pad.values[0,:,:].T, cmap='turbo', norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    vl = ax_ts.axvline(pad.time[0], c='black')

    # Animate the animated objects!
    def animate(ii):
        pc.set_array(pad.values[ii,:,:].T)
        vl.set_xdata([pad.time[ii], pad.time[ii]])
        return  

    # The animation itself, currently 0.1s cadence default
    anim = FuncAnimation(
        fig,
        animate,
        frames=kwargs.get('frames', len(pad.time)),
        interval=kwargs.get('dt', 100),
        blit=False
    )

    fig.subplots_adjust(left=0.2, right=0.9, bottom=-0.15, hspace=-0.3)
    # plt.close()

    return anim
