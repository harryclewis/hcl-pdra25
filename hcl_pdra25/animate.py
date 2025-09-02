"""
hcl_pdra25: animate.py

Contains animation functions.
"""


import astropy.units as u
from astropy.visualization import ImageNormalize, PowerStretch
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


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